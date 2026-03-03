use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;
use warp::{Filter, Reply};

mod improved_response;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Path to Codex auth.json file
    #[arg(long, default_value = "~/.codex/auth.json")]
    auth_path: String,
}

/// Chat Completions API format (what CLINE sends)
#[derive(Deserialize, Debug)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    stream: Option<bool>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
}

#[derive(Deserialize, Debug)]
struct ChatMessage {
    role: String,
    content: Value, // string or array
}

/// Chat Completions API response format
#[derive(Serialize, Debug)]
struct ChatCompletionsResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Serialize, Debug)]
struct Choice {
    index: i32,
    message: ChatResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Serialize, Debug)]
struct ChatResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Debug)]
struct Usage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

#[derive(Serialize, Debug)]
struct ResponsesApiRequest {
    model: String,
    instructions: String,
    input: Vec<ResponseItem>,
    tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "std::option::Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "std::option::Option::is_none")]
    max_output_tokens: Option<i32>,
    parallel_tool_calls: bool,
    reasoning: Option<Value>,
    store: bool,
    stream: bool,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseItem {
    Message {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        role: String,
        content: Vec<ContentItem>,
    },
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentItem {
    InputText { text: String },
    OutputText { text: String },
}

#[derive(Deserialize, Debug)]
struct AuthData {
    #[serde(rename = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,
    #[serde(rename = "api_key")]
    api_key: Option<String>,
    #[serde(rename = "access_token")]
    access_token: Option<String>,
    #[serde(rename = "account_id")]
    account_id: Option<String>,
    tokens: Option<TokenData>,
}

#[derive(Deserialize, Debug, Clone)]
struct TokenData {
    access_token: String,
    account_id: String,
}

struct ProxyServer {
    client: Client,
    auth_data: AuthData,
}

impl ProxyServer {
    async fn new(auth_path: &str) -> Result<Self> {
        let auth_path = if auth_path.starts_with("~/") {
            let home = std::env::var("HOME").context("HOME environment variable not set")?;
            auth_path.replace("~", &home)
        } else {
            auth_path.to_string()
        };

        let auth_content = tokio::fs::read_to_string(&auth_path)
            .await
            .context("Failed to read auth.json")?;

        let auth_data: AuthData =
            serde_json::from_str(&auth_content).context("Failed to parse auth.json")?;

        let client = Client::builder()
            .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { client, auth_data })
    }

    fn supported_model_ids() -> Vec<&'static str> {
        vec![
            "gpt-4",
            "gpt-5",
            "gpt-5.3-codex",
            "gpt-5.3-codex-spark",
            "gpt-5.2-codex",
            "gpt-5.1-codex-max",
            "gpt-5.2",
            "gpt-5.1-codex-mini",
        ]
    }

    fn normalize_responses_model(model: &str) -> &str {
        match model {
            "gpt-5.3-codex"
            | "gpt-5.3-codex-spark"
            | "gpt-5.2-codex"
            | "gpt-5.1-codex-max"
            | "gpt-5.2"
            | "gpt-5.1-codex-mini" => "gpt-5",
            _ => model,
        }
    }

    fn access_token(&self) -> Option<String> {
        if let Some(tokens) = &self.auth_data.tokens {
            return Some(tokens.access_token.clone());
        }
        if let Some(token) = &self.auth_data.access_token {
            return Some(token.clone());
        }
        None
    }

    fn account_id(&self) -> Option<String> {
        if let Some(tokens) = &self.auth_data.tokens {
            return Some(tokens.account_id.clone());
        }
        self.auth_data.account_id.clone()
    }

    fn api_key(&self) -> Option<String> {
        self.auth_data
            .openai_api_key
            .clone()
            .or_else(|| self.auth_data.api_key.clone())
    }

    fn resolved_static_token(&self) -> Option<String> {
        self.access_token().or_else(|| self.api_key())
    }

    fn convert_chat_to_responses(&self, chat_req: &ChatCompletionsRequest) -> ResponsesApiRequest {
        let mut input = Vec::new();
        let mut system_messages = Vec::new();

        for msg in &chat_req.messages {
            let content = flatten_chat_content(&msg.content);
            if msg.role.eq_ignore_ascii_case("system") {
                system_messages.push(content);
                continue;
            }

            let content_item = match msg.role.to_lowercase().as_str() {
                "assistant" => ContentItem::OutputText { text: content },
                _ => ContentItem::InputText { text: content },
            };

            input.push(ResponseItem::Message {
                id: None,
                role: msg.role.clone(),
                content: vec![content_item],
            });
        }

        let instructions = if system_messages.is_empty() {
            String::new()
        } else {
            system_messages.join("\n")
        };

        ResponsesApiRequest {
            model: Self::normalize_responses_model(&chat_req.model).to_string(),
            instructions,
            input,
            tools: chat_req.tools.clone().unwrap_or_default(),
            tool_choice: normalize_tool_choice(chat_req.tool_choice.clone()),
            parallel_tool_calls: false,
            reasoning: None,
            temperature: None,
            max_output_tokens: None,
            store: false,
            stream: false,
        }
    }

    fn build_headers(
        &self,
        builder: reqwest::RequestBuilder,
        auth_override: Option<&str>,
    ) -> reqwest::RequestBuilder {
        let mut request_builder = builder
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("Accept-Language", "en-US,en;q=0.9")
            .header("Accept-Encoding", "gzip, deflate, br")
            .header("Referer", "https://chatgpt.com/")
            .header("Origin", "https://chatgpt.com")
            .header("Sec-Fetch-Dest", "empty")
            .header("Sec-Fetch-Mode", "cors")
            .header("Sec-Fetch-Site", "same-origin")
            .header("Cache-Control", "no-cache")
            .header("Pragma", "no-cache")
            .header("DNT", "1")
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", "codex_cli_rs")
            .header("session_id", Uuid::new_v4().to_string());

        let resolved_override = auth_override
            .and_then(parse_bearer_token)
            .filter(|token| !is_placeholder_token(token));

        let resolved_token = resolved_override
            .or_else(|| self.access_token())
            .or_else(|| self.api_key());

        if let Some(token) = resolved_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
        }

        if let Some(account_id) = self.account_id() {
            request_builder = request_builder.header("chatgpt-account-id", account_id);
        }

        request_builder
    }

    async fn request_response_body(
        &self,
        responses_req: &ResponsesApiRequest,
        auth_override: Option<&str>,
    ) -> Result<(reqwest::StatusCode, String)> {
        let backend_response = self
            .build_headers(
                self.client
                    .post("https://chatgpt.com/backend-api/codex/responses"),
                auth_override,
            )
            .json(responses_req)
            .send()
            .await
            .context("Failed to send request to ChatGPT backend")?;

        let status = backend_response.status();
        let response_text = backend_response
            .text()
            .await
            .context("Failed to read backend response")?;

        Ok((status, response_text))
    }

    async fn proxy_request(
        &self,
        chat_req: ChatCompletionsRequest,
        auth_override: Option<String>,
    ) -> Result<ChatCompletionsResponse> {
        let mut responses_req = self.convert_chat_to_responses(&chat_req);
        // ChatGPT Codex Responses endpoint appears to require stream=true in practice.
        // Keep the API semantics to caller-side non-stream while requesting streamed backend payloads.
        responses_req.stream = true;

        let (status, response_text) = self
            .request_response_body(&responses_req, auth_override.as_deref())
            .await?;

        let (status, response_text) = if status == reqwest::StatusCode::UNAUTHORIZED
            && auth_override.as_ref().is_some_and(|override_token| {
                !is_placeholder_token(override_token) && self.resolved_static_token().is_some()
            }) {
            self.request_response_body(&responses_req, None).await?
        } else {
            (status, response_text)
        };

        if !status.is_success() {
            return Err(anyhow::anyhow!(
                "Backend response failed: {} {}",
                status,
                response_text
            ));
        }

        let content = extract_text_from_payload(&response_text);
        let fallback = improved_response::generate_contextual_response(&chat_req.messages);

        let final_text = if content.trim().is_empty() {
            fallback
        } else {
            content
        };

        Ok(ChatCompletionsResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: chat_req.model,
            choices: vec![Choice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content: final_text,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }),
        })
    }

    async fn proxy_request_stream(
        &self,
        chat_req: ChatCompletionsRequest,
        auth_override: Option<String>,
    ) -> Result<String> {
        let mut responses_req = self.convert_chat_to_responses(&chat_req);
        responses_req.stream = true;

        let (status, response_text) = self
            .request_response_body(&responses_req, auth_override.as_deref())
            .await?;

        let (status, response_text) = if status == reqwest::StatusCode::UNAUTHORIZED
            && auth_override.as_ref().is_some_and(|override_token| {
                !is_placeholder_token(override_token) && self.resolved_static_token().is_some()
            }) {
            self.request_response_body(&responses_req, None).await?
        } else {
            (status, response_text)
        };

        if !status.is_success() {
            return Err(anyhow::anyhow!(
                "Backend response failed: {} {}",
                status,
                response_text
            ));
        }

        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let chunks = build_sse_chunks(&completion_id, &chat_req.model, &response_text).await?;
        Ok(chunks)
    }
}

fn parse_bearer_token(value: &str) -> Option<String> {
    let token = value.trim();
    if token.is_empty() {
        return None;
    }
    let token = token
        .strip_prefix("Bearer ")
        .or_else(|| token.strip_prefix("bearer "))
        .unwrap_or(token)
        .trim();

    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

fn is_placeholder_token(token: &str) -> bool {
    let t = token.trim().to_lowercase();
    if t.len() < 10 {
        return true;
    }

    matches!(
        t.as_str(),
        "dummy" | "test" | "placeholder" | "xxxx" | "example" | "fake" | "none" | "null"
    ) || t.contains("your")
        || t.contains("replace")
        || t.contains("<")
        || t.contains(">")
}

fn normalize_tool_choice(tool_choice: Option<Value>) -> Option<Value> {
    tool_choice.or_else(|| Some(Value::String("auto".to_string())))
}

fn flatten_chat_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|item| {
                    if let Some(s) = item.get("text").and_then(Value::as_str) {
                        Some(s.to_string())
                    } else if let Some(s) = item.as_str() {
                        Some(s.to_string())
                    } else if let Some(v) = item.get("value").and_then(Value::as_str) {
                        Some(v.to_string())
                    } else {
                        None
                    }
                })
                .collect();

            parts.join(" ")
        }
        _ => content.to_string(),
    }
}

fn append_response_fragment(buffer: &mut String, fragment: &str) -> bool {
    let fragment = fragment.trim();
    if fragment.is_empty() {
        return false;
    }

    if buffer.is_empty() {
        buffer.push_str(fragment);
        return true;
    }

    if buffer.ends_with(fragment) {
        return false;
    }

    if let Some(rest) = fragment.strip_prefix(buffer.as_str()) {
        if !rest.is_empty() {
            buffer.push_str(rest);
        }
        return !rest.is_empty();
    }

    if buffer.contains(fragment) {
        return false;
    }

    buffer.push_str(fragment);
    true
}

fn collect_text_values(value: &Value, out: &mut String) {
    match value {
        Value::String(s) => {
            if !s.is_empty() {
                out.push_str(s);
                out.push(' ');
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_text_values(item, out);
            }
        }
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("output_text") {
                if let Some(text) = map.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                    out.push(' ');
                }
            } else {
                for (k, v) in map.iter() {
                    if k == "type" {
                        continue;
                    }
                    collect_text_values(v, out);
                }
            }
        }
        _ => {}
    }
}

fn extract_text_from_event(event: &Value) -> Option<String> {
    if let Some(delta) = event.get("delta") {
        if let Some(text) = delta.as_str() {
            return Some(text.to_string());
        }
        if let Some(map_delta) = delta.as_object() {
            if let Some(text) = map_delta.get("text").and_then(Value::as_str) {
                return Some(text.to_string());
            }
            if let Some(refusal) = map_delta.get("refusal").and_then(Value::as_str) {
                return Some(refusal.to_string());
            }
        }
    }

    if let Some(text) = event.get("text").and_then(Value::as_str) {
        return Some(text.to_string());
    }

    match event.get("type").and_then(Value::as_str) {
        Some("response.output_text.delta") => {
            if let Some(map_delta) = event.get("delta").and_then(Value::as_object) {
                if let Some(text) = map_delta.get("text").and_then(Value::as_str) {
                    return Some(text.to_string());
                }
            }
        }
        Some("response.output_text.done") => {
            if let Some(item) = event.get("item").and_then(Value::as_object) {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    return Some(text.to_string());
                }
            }
        }
        Some("response.output_item.done") => {
            if let Some(item) = event.get("item") {
                let mut text = String::new();
                if let Some(content) = item.get("content") {
                    collect_text_values(content, &mut text);
                }
                let out = text.trim();
                if !out.is_empty() {
                    return Some(out.to_string());
                }
            }
        }
        Some("response.completed") => {
            if let Some(resp) = event.get("response") {
                if let Some(output) = resp.get("output") {
                    let mut text = String::new();
                    collect_text_values(output, &mut text);
                    let out = text.trim();
                    if !out.is_empty() {
                        return Some(out.to_string());
                    }
                }
            }
        }
        _ => {}
    }

    let mut text = String::new();
    collect_text_values(event, &mut text);
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn extract_text_from_payload(payload: &str) -> String {
    // SSE format (data: ... lines) and plain JSON fallback
    if payload.contains("\n") {
        let mut chunks: Vec<String> = Vec::new();

        for line in payload.lines() {
            if let Some(json_data) = line.strip_prefix("data: ") {
                if json_data == "[DONE]" {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<Value>(json_data) {
                    if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                        let text = delta.trim();
                        if !text.is_empty() && chunks.last().map_or(true, |last| last != &text) {
                            chunks.push(delta.to_string());
                        }
                        continue;
                    }
                    if let Some(text) = event.get("text").and_then(Value::as_str) {
                        if !text.trim().is_empty() {
                            chunks.push(text.to_string());
                        }
                        continue;
                    }
                    if let Some(output) = event.get("response").and_then(Value::as_object) {
                        if let Some(content) = output.get("output") {
                            let mut block = String::new();
                            collect_text_values(content, &mut block);
                            if !block.trim().is_empty() {
                                chunks.push(block.trim().to_string());
                                continue;
                            }
                        }
                    }
                    let mut block = String::new();
                    collect_text_values(&event, &mut block);
                    if !block.trim().is_empty() {
                        chunks.push(block.trim().to_string());
                    }
                }
            }
        }

        let mut merged = String::new();
        for chunk in chunks {
            append_response_fragment(&mut merged, &chunk);
        }
        if !merged.trim().is_empty() {
            return merged;
        }
    }

    if let Ok(root) = serde_json::from_str::<Value>(payload) {
        let mut out = String::new();
        if let Some(response) = root.get("response") {
            if let Some(output) = response.get("output") {
                collect_text_values(output, &mut out);
            } else {
                collect_text_values(response, &mut out);
            }
        } else if let Some(output) = root.get("output") {
            collect_text_values(output, &mut out);
        } else {
            collect_text_values(&root, &mut out);
        }

        if let Some(item) = root.get("text").and_then(Value::as_str) {
            out.push(' ');
            out.push_str(item);
        }
        return out.trim().to_string();
    }

    String::new()
}

fn build_json_chunk(
    id: &str,
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&str>,
) -> Result<String> {
    let delta = json!({
        "role": role,
        "content": content,
    });
    let choices = vec![json!({
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
    })];

    let payload = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": Utc::now().timestamp(),
        "model": model,
        "choices": choices,
    });

    Ok(format!(
        "data: {}\n\n",
        serde_json::to_string(&payload).context("failed to serialize stream chunk")?
    ))
}

async fn build_sse_chunks(completion_id: &str, model: &str, payload: &str) -> Result<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut collected: String = String::new();

    chunks.push(build_json_chunk(
        completion_id,
        model,
        Some("assistant"),
        None,
        None,
    )?);

    let is_sse = payload.lines().any(|line| line.starts_with("data: "));

    if !is_sse {
        // non-sse backend response fallback
        let text = extract_text_from_payload(payload);
        if !text.is_empty() {
            chunks.push(build_json_chunk(
                completion_id,
                model,
                None,
                Some(&escape_sse_content(&text)),
                None,
            )?);
        }
    } else {
        for line in payload.lines() {
            if let Some(json_data) = line.strip_prefix("data: ") {
                if json_data == "[DONE]" {
                    break;
                }
                if json_data.trim().is_empty() {
                    continue;
                }

                if let Ok(event) = serde_json::from_str::<Value>(json_data) {
                    if let Some(event_type) = event.get("type").and_then(Value::as_str) {
                        if matches!(
                            event_type,
                            "response.failed"
                                | "response.output_text.done"
                                | "response.output_item.done"
                                | "response.completed"
                        ) {
                            continue;
                        }
                    }

                    if let Some(event_text) = extract_text_from_event(&event) {
                        let event_text = event_text.trim();
                        if append_response_fragment(&mut collected, event_text) {
                            chunks.push(build_json_chunk(
                                completion_id,
                                model,
                                None,
                                Some(&escape_sse_content(event_text)),
                                None,
                            )?);
                        }
                    }
                }
            }
        }
    }

    if collected.is_empty() {
        let fallback = improved_response::generate_contextual_response(&[]);
        chunks.push(build_json_chunk(
            completion_id,
            model,
            None,
            Some(&fallback),
            None,
        )?);
    }

    chunks.push(build_json_chunk(
        completion_id,
        model,
        None,
        Some(""),
        Some("stop"),
    )?);
    chunks.push("data: [DONE]\n\n".to_string());

    Ok(chunks.concat())
}

fn escape_sse_content(value: &str) -> String {
    // Keep SSE chunk payload valid JSON text, then json! macro escapes for us.
    value.to_string()
}

// Enhanced logging function
fn log_request(method: &warp::http::Method, path: &str, headers: &warp::http::HeaderMap) {
    let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S%.3f UTC");

    println!("\n🔍 === INTERCEPTED REQUEST ===");
    println!("⏰ Timestamp: {}", timestamp);
    println!("📥 Method: {}", method);
    println!("📍 Path: {}", path);

    println!("\n📋 Headers ({} total):", headers.len());
    for (name, value) in headers.iter() {
        let header_name = name.as_str().to_lowercase();
        let value_str = match value.to_str() {
            Ok(v) => v,
            Err(_) => "[INVALID UTF-8]",
        };

        if header_name.contains("user-agent")
            || header_name.contains("client")
            || header_name.contains("cline")
        {
            println!("  🎯 {}: {}", name, value_str);
        } else if header_name == "authorization" {
            println!(
                "  🔐 {}: {}***",
                name,
                &value_str[..std::cmp::min(20, value_str.len())]
            );
        } else {
            println!("  📄 {}: {}", name, value_str);
        }
    }

    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("none");
    if user_agent.to_lowercase().contains("vscode") {
        println!("🎯 DETECTED: VS Code client!");
    }
    if user_agent.to_lowercase().contains("cline") {
        println!("🎯 DETECTED: CLINE extension!");
    }

    println!("🔍 === END INTERCEPT ===\n");
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Initializing Codex OpenAI Proxy...");

    let proxy = ProxyServer::new(&args.auth_path).await?;
    println!("✓ Loaded authentication from {}", args.auth_path);

    let proxy_filter = warp::any().map(move || proxy.clone());

    let universal_handler = warp::any()
        .and(warp::method())
        .and(warp::path::full())
        .and(warp::header::headers_cloned())
        .and(warp::body::bytes())
        .and(proxy_filter.clone())
        .and_then(universal_request_handler);

    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec![
            "authorization",
            "content-type",
            "accept",
            "accept-encoding",
            "x-stainless-arch",
            "x-stainless-lang",
            "x-stainless-os",
            "x-stainless-package-version",
            "x-stainless-retry-count",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
            "x-stainless-timeout",
        ])
        .allow_methods(vec!["GET", "POST", "PUT", "DELETE", "OPTIONS"]);

    println!(
        "🚀 Codex OpenAI Proxy listening on http://0.0.0.0:{}",
        args.port
    );
    println!("   Health check: http://localhost:{}/health", args.port);
    println!(
        "   Chat endpoint: http://localhost:{}/v1/chat/completions",
        args.port
    );

    warp::serve(universal_handler.with(cors).with(warp::log("codex_proxy")))
        .run(([0, 0, 0, 0], args.port))
        .await;

    Ok(())
}

async fn universal_request_handler(
    method: warp::http::Method,
    path: warp::path::FullPath,
    headers: warp::http::HeaderMap,
    body: bytes::Bytes,
    proxy: ProxyServer,
) -> Result<impl Reply, warp::Rejection> {
    let path_str = path.as_str();
    let normalized_path = path_str
        .split('?')
        .next()
        .unwrap_or("")
        .trim_end_matches('/');
    let normalized_path = if normalized_path.is_empty() {
        "/"
    } else {
        normalized_path
    };
    log_request(&method, path_str, &headers);

    match (method.as_str(), normalized_path) {
        ("GET", "/health") => Ok(warp::reply::json(
            &json!({"status": "ok", "service": "codex-openai-proxy"}),
        )
        .into_response()),
        ("GET", "/models") | ("GET", "/v1/models") => {
            let now = Utc::now().timestamp();
            let models: Vec<Value> = ProxyServer::supported_model_ids()
                .into_iter()
                .map(|id| {
                    json!({
                        "id": id,
                        "object": "model",
                        "created": now,
                        "owned_by": "openai"
                    })
                })
                .collect();
            let models_response = json!({
                "object": "list",
                "data": models,
            });
            Ok(warp::reply::json(&models_response).into_response())
        }
        ("POST", "/chat/completions") | ("POST", "/v1/chat/completions") => {
            let chat_req: ChatCompletionsRequest = match serde_json::from_slice(&body) {
                Ok(req) => req,
                Err(e) => {
                    println!("❌ JSON parse error: {}", e);
                    return Ok(warp::reply::with_status(
                        "Invalid JSON",
                        warp::http::StatusCode::BAD_REQUEST,
                    )
                    .into_response());
                }
            };

            if chat_req.stream.unwrap_or(false) {
                let auth_header = headers
                    .get("authorization")
                    .and_then(|value| value.to_str().ok())
                    .map(|value| value.to_string());

                match proxy.proxy_request_stream(chat_req, auth_header).await {
                    Ok(sse) => {
                        let reply =
                            warp::reply::with_header(sse, "content-type", "text/event-stream");
                        let reply = warp::reply::with_header(reply, "cache-control", "no-cache");
                        let reply = warp::reply::with_header(reply, "connection", "keep-alive");
                        let reply =
                            warp::reply::with_header(reply, "access-control-allow-origin", "*");
                        Ok(reply.into_response())
                    }
                    Err(e) => {
                        println!("❌ Stream proxy error: {:#}", e);
                        let fallback = improved_response::generate_contextual_response(&[]);
                        let chunk_id = format!("chatcmpl-{}", Uuid::new_v4());
                        let message_chunk = json!({
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": Utc::now().timestamp(),
                            "model": "gpt-5",
                            "choices": [{
                                "index": 0,
                                "delta": { "role": "assistant", "content": fallback },
                                "finish_reason": null
                            }]
                        });
                        let final_chunk = "data: [DONE]\n\n";
                        let sse = format!(
                            "data: {}\n\n{}",
                            serde_json::to_string(&message_chunk)
                                .unwrap_or_else(|_| "{}".to_string()),
                            final_chunk
                        );
                        let reply =
                            warp::reply::with_header(sse, "content-type", "text/event-stream");
                        let reply =
                            warp::reply::with_header(reply, "access-control-allow-origin", "*");
                        Ok(reply.into_response())
                    }
                }
            } else {
                let auth_header = headers
                    .get("authorization")
                    .and_then(|value| value.to_str().ok())
                    .map(|value| value.to_string());

                match proxy.proxy_request(chat_req, auth_header).await {
                    Ok(response) => {
                        let reply = warp::reply::json(&response);
                        let reply =
                            warp::reply::with_header(reply, "content-type", "application/json");
                        let reply =
                            warp::reply::with_header(reply, "access-control-allow-origin", "*");
                        Ok(reply.into_response())
                    }
                    Err(e) => {
                        eprintln!("Proxy error: {:#}", e);
                        let reply = warp::reply::json(&json!({
                            "error": {
                                "message": format!("Proxy error: {}", e),
                                "type": "proxy_error",
                                "code": "internal_error"
                            }
                        }));
                        let reply =
                            warp::reply::with_status(reply, warp::http::StatusCode::BAD_GATEWAY);
                        let reply =
                            warp::reply::with_header(reply, "content-type", "application/json");
                        let reply =
                            warp::reply::with_header(reply, "access-control-allow-origin", "*");
                        Ok(reply.into_response())
                    }
                }
            }
        }
        _ => Ok(
            warp::reply::with_status("Not found", warp::http::StatusCode::NOT_FOUND)
                .into_response(),
        ),
    }
}

// Make ProxyServer cloneable for warp filters
impl Clone for ProxyServer {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            auth_data: AuthData {
                openai_api_key: self.auth_data.openai_api_key.clone(),
                api_key: self.auth_data.api_key.clone(),
                access_token: self.auth_data.access_token.clone(),
                account_id: self.auth_data.account_id.clone(),
                tokens: self.auth_data.tokens.clone(),
            },
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bearer_token_supports_various_prefixes() {
        assert_eq!(
            parse_bearer_token("Bearer sk-test-123"),
            Some("sk-test-123".to_string())
        );
        assert_eq!(
            parse_bearer_token("bearer sk-test-456"),
            Some("sk-test-456".to_string())
        );
        assert_eq!(
            parse_bearer_token("  token-without-prefix  "),
            Some("token-without-prefix".to_string())
        );
        assert_eq!(parse_bearer_token(""), None);
        assert_eq!(parse_bearer_token("   "), None);
    }

    #[test]
    fn flatten_chat_content_works_with_text_and_objects() {
        let content = json!([
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
            "bye"
        ]);
        let flat = flatten_chat_content(&content);
        assert_eq!(flat, "hello world bye");
    }

    #[test]
    fn convert_chat_to_responses_uses_output_text_for_assistant_role() {
        let auth_data = AuthData {
            openai_api_key: None,
            api_key: Some("sk-test".to_string()),
            access_token: None,
            account_id: None,
            tokens: None,
        };
        let proxy = ProxyServer {
            client: Client::new(),
            auth_data,
        };
        let chat_req = ChatCompletionsRequest {
            model: "gpt-5.3-codex-spark".to_string(),
            messages: vec![
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Value::String("reply in Chinese".to_string()),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Value::String("Hello".to_string()),
                },
            ],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
        };

        let responses = proxy.convert_chat_to_responses(&chat_req);
        match &responses.input[0] {
            ResponseItem::Message { content, .. } => match &content[0] {
                ContentItem::OutputText { text } => assert_eq!(text, "reply in Chinese"),
                _ => panic!("assistant message should use OutputText"),
            },
        }
        match &responses.input[1] {
            ResponseItem::Message { content, .. } => match &content[0] {
                ContentItem::InputText { text } => assert_eq!(text, "Hello"),
                _ => panic!("user message should use InputText"),
            },
        }
    }

    #[test]
    fn convert_chat_to_responses_moves_system_messages_into_instructions() {
        let auth_data = AuthData {
            openai_api_key: None,
            api_key: Some("sk-test".to_string()),
            access_token: None,
            account_id: None,
            tokens: None,
        };
        let proxy = ProxyServer {
            client: Client::new(),
            auth_data,
        };
        let chat_req = ChatCompletionsRequest {
            model: "gpt-5.3-codex-spark".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: Value::String("Always speak in one sentence.".to_string()),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Value::String("hello".to_string()),
                },
            ],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
        };

        let responses = proxy.convert_chat_to_responses(&chat_req);
        assert_eq!(responses.model, "gpt-5");
        assert_eq!(responses.instructions, "Always speak in one sentence.");
        assert_eq!(responses.input.len(), 1);
        match &responses.input[0] {
            ResponseItem::Message { role, content, .. } => {
                assert_eq!(role, "user");
                assert_eq!(content.len(), 1);
            }
        }
    }

    #[test]
    fn is_placeholder_token_rejects_too_short_and_dummy() {
        assert!(is_placeholder_token("dummy"));
        assert!(is_placeholder_token("   "));
        assert!(is_placeholder_token("short"));
        assert!(!is_placeholder_token(
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.example.longtoken"
        ));
        assert!(!is_placeholder_token(
            "sk-very-long-actual-like-openai-token-1234567890"
        ));
    }

    #[test]
    fn collect_text_values_output_text_does_not_duplicate_text() {
        let raw = json!({
            "type": "output_text",
            "text": "hi",
            "annotations": [
                {
                    "type": "url_citation",
                    "url": "https://example.com"
                }
            ]
        });

        let mut out = String::new();
        collect_text_values(&raw, &mut out);
        assert_eq!(out.trim(), "hi");
    }

    #[test]
    fn normalize_responses_model_aliases_gpt5_codex_variants_to_gpt5() {
        assert_eq!(
            ProxyServer::normalize_responses_model("gpt-5.3-codex"),
            "gpt-5"
        );
        assert_eq!(
            ProxyServer::normalize_responses_model("gpt-5.3-codex-spark"),
            "gpt-5"
        );
        assert_eq!(ProxyServer::normalize_responses_model("gpt-5"), "gpt-5");
        assert_eq!(ProxyServer::normalize_responses_model("gpt-4"), "gpt-4");
    }

    #[test]
    fn extract_text_from_event_handles_output_item_done_and_completed() {
        let output_item_done = json!({
            "type": "response.output_item.done",
            "item": {
                "content": [
                    {"type": "output_text", "text": "hello"},
                    " world"
                ]
            }
        });
        let item_text = extract_text_from_event(&output_item_done).expect("text");
        assert!(item_text.contains("hello"));

        let output_text_delta = json!({
            "type": "response.output_text.delta",
            "delta": {
                "text": "delta text"
            }
        });
        assert_eq!(
            extract_text_from_event(&output_text_delta).expect("text"),
            "delta text"
        );

        let completed = json!({
            "type": "response.completed",
            "response": {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "done"}]}
                ]
            }
        });
        assert_eq!(extract_text_from_event(&completed).expect("text"), "done");
    }

    #[tokio::test]
    async fn build_sse_chunks_handles_response_output_events() {
        let payload = [
            "data: {\"type\": \"response.output_text.delta\", \"delta\": {\"text\": \"Hello\"}}",
            "data: {\"type\": \"response.output_text.delta\", \"delta\": {\"text\": \" world\"}}",
            "data: [DONE]",
        ]
        .join("\n");

        let sse = build_sse_chunks("chatcmpl-test", "gpt-5", &payload)
            .await
            .expect("build sse");

        assert!(sse.contains("Hello"));
        assert!(sse.contains("world"));
        assert!(sse.ends_with("data: [DONE]\n\n"));
    }
}
