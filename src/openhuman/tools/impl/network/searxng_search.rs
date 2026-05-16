//! SearXNG search tool: calls a self-hosted SearXNG instance's `/search?format=json`
//! endpoint and normalises results into the same markdown shape used by `web_search`.
//!
//! # SSRF policy divergence from `url_guard`
//!
//! The standard `validate_url` helper (used by `web_fetch` and `http_request`) blocks
//! loopback/private addresses to prevent SSRF. For SearXNG we deliberately allow
//! loopback (e.g. `http://localhost:8080`) because SearXNG is commonly self-hosted on
//! the same machine as the OpenHuman core. The user has already explicitly opted in by
//! setting `searxng.enabled = true` and providing `base_url` in config; that explicit
//! opt-in is equivalent to granting intent for a loopback target. Private-network IPs
//! beyond loopback (10.x, 172.16.x, 192.168.x) are also allowed for the same reason:
//! operators often run SearXNG on a local network host.
//!
//! We still reject: IPv6 bracket notation, userinfo (`user@host`), non-http(s) schemes,
//! and empty/whitespace URLs.

use crate::openhuman::tools::traits::{Tool, ToolCallOptions, ToolResult};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Duration;

// ---------------------------------------------------------------------------
// SearXNG response shape
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearxngResponse {
    #[serde(default)]
    pub results: Vec<SearxngResult>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub number_of_results: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct SearxngResult {
    pub title: Option<String>,
    pub url: String,
    pub content: Option<String>,
    pub engine: Option<String>,
    #[serde(rename = "publishedDate")]
    pub published_date: Option<String>,
    pub score: Option<f64>,
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

pub struct SearxngSearchTool {
    /// Base URL of the SearXNG instance, e.g. "http://localhost:8080".
    base_url: String,
    max_results: usize,
    default_language: String,
    default_categories: String,
    timeout_secs: u64,
}

impl SearxngSearchTool {
    pub fn new(
        base_url: String,
        max_results: usize,
        default_language: String,
        default_categories: String,
        timeout_secs: u64,
    ) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            max_results: max_results.clamp(1, 50),
            default_language,
            default_categories,
            timeout_secs: timeout_secs.max(1),
        }
    }

    fn validate_base_url(url: &str) -> anyhow::Result<()> {
        let url = url.trim();
        if url.is_empty() {
            anyhow::bail!("SearXNG base_url cannot be empty");
        }
        if url.chars().any(char::is_whitespace) {
            anyhow::bail!("SearXNG base_url cannot contain whitespace");
        }
        if !url.starts_with("http://") && !url.starts_with("https://") {
            anyhow::bail!("SearXNG base_url must start with http:// or https://");
        }
        // Extract authority portion to check for userinfo/IPv6.
        let rest = url
            .strip_prefix("http://")
            .or_else(|| url.strip_prefix("https://"))
            .unwrap_or("");
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        if authority.contains('@') {
            anyhow::bail!("SearXNG base_url must not include userinfo (user:pass@host)");
        }
        if authority.starts_with('[') {
            anyhow::bail!("IPv6 hosts are not supported in searxng.base_url");
        }
        Ok(())
    }

    fn parse_categories(&self, raw: Option<&str>) -> String {
        let cats = raw.unwrap_or_default().trim();
        if cats.is_empty() {
            return self.default_categories.clone();
        }
        // Validate against known SearXNG categories; fall back to default on unknown.
        const VALID: &[&str] = &[
            "general",
            "news",
            "images",
            "videos",
            "science",
            "social_media",
            "files",
        ];
        let parts: Vec<&str> = cats
            .split(',')
            .map(str::trim)
            .filter(|s| VALID.contains(s))
            .collect();
        if parts.is_empty() {
            return self.default_categories.clone();
        }
        parts.join(",")
    }

    fn format_results(&self, results: &[SearxngResult], query: &str, max_results: usize) -> String {
        if results.is_empty() {
            return format!("No results found for: {}", query);
        }

        let mut lines = vec![format!("Search results for: {} (via SearXNG)", query)];

        for (i, result) in results.iter().take(max_results).enumerate() {
            let title = result
                .title
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("No title");

            lines.push(format!("{}. {}", i + 1, title));
            lines.push(format!("   {}", result.url.trim()));

            if let Some(date) = result.published_date.as_deref() {
                let date = date.trim();
                if !date.is_empty() {
                    lines.push(format!("   Published: {}", date));
                }
            }

            if let Some(content) = result.content.as_deref() {
                let snippet = content.trim();
                if !snippet.is_empty() {
                    let truncated = crate::openhuman::util::truncate_with_ellipsis(snippet, 500);
                    lines.push(format!("   {}", truncated));
                }
            }
        }

        lines.join("\n")
    }

    fn render_results_markdown(
        &self,
        results: &[SearxngResult],
        query: &str,
        max_results: usize,
    ) -> String {
        if results.is_empty() {
            return format!("_No results for `{query}`._");
        }
        let mut out = format!("# Search results: `{query}` (SearXNG)\n");
        for r in results.iter().take(max_results) {
            let title = r
                .title
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("Untitled");
            out.push_str(&format!("\n## [{title}]({})\n", r.url.trim()));
            if let Some(date) = r.published_date.as_deref() {
                let date = date.trim();
                if !date.is_empty() {
                    out.push_str(&format!("_Published: {date}_\n\n"));
                }
            }
            if let Some(content) = r.content.as_deref() {
                let snippet = content.trim();
                if !snippet.is_empty() {
                    let truncated =
                        crate::openhuman::util::truncate_with_suffix(snippet, 500, "...");
                    out.push_str(&format!("> {truncated}\n"));
                }
            }
        }
        out
    }
}

#[async_trait]
impl Tool for SearxngSearchTool {
    fn name(&self) -> &str {
        "searxng_search_tool"
    }

    fn description(&self) -> &str {
        "Search the web via a self-hosted SearXNG instance. Returns results with titles, \
         URLs, and snippets. Supports categories (general, news, images, videos, science, \
         social_media, files) and language filtering. Requires a configured SearXNG base_url."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific for better results."
                },
                "categories": {
                    "type": "string",
                    "description": "Comma-separated SearXNG categories. Valid values: general, news, images, videos, science, social_media, files. Defaults to config value."
                },
                "language": {
                    "type": "string",
                    "description": "Language code for results, e.g. \"en\", \"de\". Defaults to config value."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (1-50). Defaults to config value.",
                    "minimum": 1,
                    "maximum": 50
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.execute_with_options(args, ToolCallOptions::default())
            .await
    }

    fn supports_markdown(&self) -> bool {
        true
    }

    async fn execute_with_options(
        &self,
        args: serde_json::Value,
        options: ToolCallOptions,
    ) -> anyhow::Result<ToolResult> {
        let start = std::time::Instant::now();

        let query = args
            .get("query")
            .and_then(|q| q.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing or empty required parameter: query"))?;

        let categories = self.parse_categories(args.get("categories").and_then(|v| v.as_str()));

        let language = args
            .get("language")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| self.default_language.as_str())
            .to_string();

        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .map(|n| (n as usize).clamp(1, 50))
            .unwrap_or(self.max_results);

        let query_fingerprint = hex::encode(Sha256::digest(query.as_bytes()));
        tracing::debug!(
            query_len = query.chars().count(),
            query_fingerprint = %&query_fingerprint[..16],
            categories = %categories,
            language = %language,
            max_results,
            timeout_secs = self.timeout_secs,
            "[searxng_search] starting request"
        );

        // Validate the stored base_url before every call (cheap, catches misconfiguration early).
        Self::validate_base_url(&self.base_url)?;

        let search_url = format!("{}/search", self.base_url);

        let client = Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .build()?;

        tracing::debug!(
            url = %search_url,
            categories = %categories,
            language = %language,
            max_results,
            "[searxng_search] sending GET request"
        );

        let response = client
            .get(&search_url)
            .query(&[
                ("q", query),
                ("format", "json"),
                ("categories", categories.as_str()),
                ("language", language.as_str()),
                ("pageno", "1"),
            ])
            .send()
            .await
            .map_err(|e| {
                tracing::debug!(
                    error = %e,
                    elapsed_ms = start.elapsed().as_millis(),
                    "[searxng_search] request failed"
                );
                anyhow::anyhow!("SearXNG request failed: {e}")
            })?;

        let status = response.status();
        tracing::debug!(
            status_code = status.as_u16(),
            elapsed_ms = start.elapsed().as_millis(),
            "[searxng_search] received response"
        );

        if !status.is_success() {
            anyhow::bail!(
                "SearXNG returned non-2xx status: {} (check that the instance is running and the base_url is correct)",
                status
            );
        }

        let searxng_resp: SearxngResponse = response.json().await.map_err(|e| {
            tracing::debug!(
                error = %e,
                elapsed_ms = start.elapsed().as_millis(),
                "[searxng_search] failed to parse response JSON"
            );
            anyhow::anyhow!("Failed to parse SearXNG response: {e}")
        })?;

        let result_count = searxng_resp.results.len();
        tracing::debug!(
            result_count,
            elapsed_ms = start.elapsed().as_millis(),
            "[searxng_search] complete"
        );

        let mut tool_result =
            ToolResult::success(self.format_results(&searxng_resp.results, query, max_results));
        if options.prefer_markdown {
            tool_result.markdown_formatted =
                Some(self.render_results_markdown(&searxng_resp.results, query, max_results));
        }

        Ok(tool_result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::Query, routing::get, Json, Router};
    use serde::Deserialize as De;
    use serde_json::json;

    fn tool() -> SearxngSearchTool {
        SearxngSearchTool::new(
            "http://127.0.0.1:8080".into(),
            10,
            "en".into(),
            "general".into(),
            10,
        )
    }

    async fn start_mock_searxng(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    // ---- unit tests for query construction / parsing ----

    #[test]
    fn test_tool_name() {
        assert_eq!(tool().name(), "searxng_search_tool");
    }

    #[test]
    fn test_tool_description_mentions_searxng() {
        assert!(tool().description().contains("SearXNG"));
    }

    #[test]
    fn test_parameters_schema_has_required_query() {
        let schema = tool().parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["query"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("query")));
    }

    #[test]
    fn test_parse_categories_defaults_on_empty() {
        let t = SearxngSearchTool::new(
            "http://localhost:8080".into(),
            5,
            "en".into(),
            "general".into(),
            10,
        );
        assert_eq!(t.parse_categories(None), "general");
        assert_eq!(t.parse_categories(Some("")), "general");
    }

    #[test]
    fn test_parse_categories_passes_valid() {
        let t = tool();
        assert_eq!(t.parse_categories(Some("news")), "news");
        assert_eq!(t.parse_categories(Some("news,science")), "news,science");
    }

    #[test]
    fn test_parse_categories_filters_invalid() {
        let t = tool();
        // "movies" is not a valid SearXNG category in our allowlist
        assert_eq!(
            t.parse_categories(Some("news,movies")),
            "news",
            "invalid category should be stripped"
        );
    }

    #[test]
    fn test_parse_categories_all_invalid_falls_back() {
        let t = tool();
        assert_eq!(t.parse_categories(Some("movies,audio")), "general");
    }

    // ---- response parsing ----

    #[test]
    fn test_format_results_empty() {
        let result = tool().format_results(&[], "test query", 10);
        assert!(result.contains("No results found"));
    }

    #[test]
    fn test_format_results_with_data() {
        let results = vec![
            SearxngResult {
                title: Some("SearXNG Docs".into()),
                url: "https://docs.searxng.org".into(),
                content: Some("SearXNG is a free meta-search engine.".into()),
                engine: Some("duckduckgo".into()),
                published_date: None,
                score: Some(1.0),
            },
            SearxngResult {
                title: Some("GitHub SearXNG".into()),
                url: "https://github.com/searxng/searxng".into(),
                content: None,
                engine: Some("google".into()),
                published_date: Some("2024-01-15".into()),
                score: Some(0.9),
            },
        ];

        let result = tool().format_results(&results, "searxng", 10);
        assert!(result.contains("via SearXNG"));
        assert!(result.contains("SearXNG Docs"));
        assert!(result.contains("https://docs.searxng.org"));
        assert!(result.contains("SearXNG is a free meta-search engine."));
        assert!(result.contains("Published: 2024-01-15"));
    }

    #[test]
    fn test_format_results_respects_max_results() {
        let t = SearxngSearchTool::new(
            "http://localhost:8080".into(),
            2,
            "en".into(),
            "general".into(),
            10,
        );
        let results: Vec<SearxngResult> = (1..=5)
            .map(|i| SearxngResult {
                title: Some(format!("Result {i}")),
                url: format!("https://example.com/{i}"),
                content: None,
                engine: None,
                published_date: None,
                score: None,
            })
            .collect();

        let output = t.format_results(&results, "q", 2);
        assert!(output.contains("Result 1"));
        assert!(output.contains("Result 2"));
        assert!(!output.contains("Result 3"));
    }

    #[test]
    fn test_format_results_handles_missing_title() {
        let results = vec![SearxngResult {
            title: None,
            url: "https://example.com".into(),
            content: Some("Some content".into()),
            engine: None,
            published_date: None,
            score: None,
        }];
        let output = tool().format_results(&results, "q", 10);
        assert!(output.contains("No title"));
    }

    #[test]
    fn test_format_results_truncates_long_content() {
        let long_content = "x".repeat(600);
        let results = vec![SearxngResult {
            title: Some("T".into()),
            url: "https://example.com".into(),
            content: Some(long_content),
            engine: None,
            published_date: None,
            score: None,
        }];
        let output = tool().format_results(&results, "q", 10);
        assert!(output.contains("..."));
        let line = output.lines().find(|l| l.trim().starts_with('x')).unwrap();
        // 500 chars + "..." = 503 at most, plus the "   " indent prefix
        assert!(line.trim().len() <= 503);
    }

    // ---- validation ----

    #[test]
    fn test_validate_base_url_accepts_http_localhost() {
        assert!(SearxngSearchTool::validate_base_url("http://localhost:8080").is_ok());
    }

    #[test]
    fn test_validate_base_url_accepts_https_public() {
        assert!(SearxngSearchTool::validate_base_url("https://search.example.com").is_ok());
    }

    #[test]
    fn test_validate_base_url_rejects_empty() {
        assert!(SearxngSearchTool::validate_base_url("").is_err());
    }

    #[test]
    fn test_validate_base_url_rejects_ftp() {
        assert!(SearxngSearchTool::validate_base_url("ftp://host").is_err());
    }

    #[test]
    fn test_validate_base_url_rejects_userinfo() {
        assert!(SearxngSearchTool::validate_base_url("http://user@host:8080").is_err());
    }

    #[test]
    fn test_validate_base_url_rejects_ipv6() {
        assert!(SearxngSearchTool::validate_base_url("http://[::1]:8080").is_err());
    }

    #[test]
    fn test_validate_base_url_rejects_whitespace() {
        assert!(SearxngSearchTool::validate_base_url("http://host /path").is_err());
    }

    // ---- async error handling ----

    #[tokio::test]
    async fn test_execute_missing_query() {
        let result = tool().execute(json!({})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("query"));
    }

    #[tokio::test]
    async fn test_execute_empty_query() {
        let result = tool().execute(json!({"query": "  "})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_returns_error_on_non_2xx() {
        #[derive(De)]
        struct Params {
            q: String,
            format: String,
        }

        let app = Router::new().route(
            "/search",
            get(|Query(_p): Query<Params>| async move {
                (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "service unavailable",
                )
            }),
        );

        let base_url = start_mock_searxng(app).await;
        let t = SearxngSearchTool::new(base_url, 5, "en".into(), "general".into(), 5);
        let result = t.execute(json!({"query": "test"})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-2xx status"));
    }

    #[tokio::test]
    async fn test_execute_returns_error_on_malformed_json() {
        let app = Router::new().route("/search", get(|| async { "not-valid-json" }));

        let base_url = start_mock_searxng(app).await;
        let t = SearxngSearchTool::new(base_url, 5, "en".into(), "general".into(), 5);
        let result = t.execute(json!({"query": "test"})).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("parse SearXNG response"));
    }

    #[tokio::test]
    async fn test_execute_success_full_results() {
        use std::sync::{Arc, Mutex};
        let called = Arc::new(Mutex::new(false));
        let called_clone = Arc::clone(&called);

        #[derive(De)]
        struct Params {
            q: Option<String>,
            format: Option<String>,
            categories: Option<String>,
            language: Option<String>,
        }

        let app = Router::new().route(
            "/search",
            get(move |Query(p): Query<Params>| {
                let called_inner = Arc::clone(&called_clone);
                async move {
                    // Verify the query params were forwarded.
                    assert_eq!(p.format.as_deref(), Some("json"));
                    assert!(p.q.is_some());
                    *called_inner.lock().unwrap() = true;
                    Json(json!({
                        "query": p.q,
                        "number_of_results": 1,
                        "results": [
                            {
                                "title": "SearXNG Home",
                                "url": "https://searxng.org",
                                "content": "Privacy-respecting metasearch engine.",
                                "engine": "duckduckgo",
                                "publishedDate": "2024-06-01",
                                "score": 1.0
                            }
                        ]
                    }))
                }
            }),
        );

        let base_url = start_mock_searxng(app).await;
        let t = SearxngSearchTool::new(base_url, 5, "en".into(), "general".into(), 10);

        let result = t
            .execute(json!({ "query": "searxng privacy" }))
            .await
            .expect("execute should succeed with valid mock");

        assert!(*called.lock().unwrap());
        assert!(result.output().contains("SearXNG Home"));
        assert!(result.output().contains("https://searxng.org"));
        assert!(result
            .output()
            .contains("Privacy-respecting metasearch engine."));
        assert!(result.output().contains("Published: 2024-06-01"));
    }

    #[tokio::test]
    async fn test_execute_success_empty_results() {
        let app = Router::new().route(
            "/search",
            get(|| async {
                Json(json!({
                    "query": "obscure query",
                    "results": []
                }))
            }),
        );

        let base_url = start_mock_searxng(app).await;
        let t = SearxngSearchTool::new(base_url, 5, "en".into(), "general".into(), 10);

        let result = t
            .execute(json!({"query": "obscure query"}))
            .await
            .expect("empty results should succeed");

        assert!(result.output().contains("No results found"));
    }

    #[tokio::test]
    async fn test_execute_respects_max_results_param() {
        let app = Router::new().route(
            "/search",
            get(|| async {
                // Return 10 results; tool should honour max_results=3 from call arg.
                let results: Vec<serde_json::Value> = (1..=10)
                    .map(|i| {
                        json!({
                            "title": format!("Result {i}"),
                            "url": format!("https://example.com/{i}"),
                            "content": null,
                            "engine": "test"
                        })
                    })
                    .collect();
                Json(json!({ "results": results }))
            }),
        );

        let base_url = start_mock_searxng(app).await;
        let t = SearxngSearchTool::new(
            base_url,
            5, // default max_results = 5
            "en".into(),
            "general".into(),
            10,
        );

        let result = t
            .execute(json!({"query": "test", "max_results": 3}))
            .await
            .expect("should succeed");

        let output = result.output();
        assert!(output.contains("Result 1"));
        assert!(output.contains("Result 2"));
        assert!(output.contains("Result 3"));
        // Results 4+ should be trimmed.
        assert!(!output.contains("Result 4"));
    }

    #[tokio::test]
    async fn test_execute_markdown_output() {
        let app = Router::new().route(
            "/search",
            get(|| async {
                Json(json!({
                    "results": [{
                        "title": "Test Page",
                        "url": "https://test.example.com",
                        "content": "A short snippet.",
                        "engine": "google"
                    }]
                }))
            }),
        );

        let base_url = start_mock_searxng(app).await;
        let t = SearxngSearchTool::new(base_url, 5, "en".into(), "general".into(), 10);

        let result = t
            .execute_with_options(
                json!({"query": "test"}),
                ToolCallOptions {
                    prefer_markdown: true,
                    ..Default::default()
                },
            )
            .await
            .expect("markdown execute should succeed");

        let md = result
            .markdown_formatted
            .as_deref()
            .expect("markdown_formatted should be set");
        assert!(md.contains("## [Test Page]"));
        assert!(md.contains("A short snippet."));
    }
}
