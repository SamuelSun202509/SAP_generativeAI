from __future__ import annotations
import requests
import re
import time
from typing import Optional, Tuple
from bs4 import BeautifulSoup
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain.tools import tool
from langchain.agents import create_agent


class WebURLDetailAgent:
    """
    Reusable class that builds a LangChain-based multi-agent pipeline:
      - Article tool: fetch & extract article text (preserve paragraphs)
      - Translation tool: translate text into target language (default: Chinese)
      - Image tool: describe image content via vision-capable LLM
      - Two sub-agents (article/image) wrapped as tools
      - One supervisor agent to route based on URL
    
    Public method:
      - run(url: str, target_language: Optional[str] = None) -> str
    """

    # ------------ Defaults ------------
    DEFAULT_MODEL_NAME = "gpt-4o"
    DEFAULT_TEMPERATURE = 0
    DEFAULT_MAX_TOKENS = 12_000
    DEFAULT_TARGET_LANGUAGE = "Chinese"
    DEFAULT_MAX_CHARS = 10_000
    DEFAULT_HTTP_TIMEOUT = 15
    DEFAULT_RETRIES = 2
    DEFAULT_BACKOFF = 0.8

    # Common image extensions for quick heuristics
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg")

    def __init__(
        self,
        model_name: str = None,
        temperature: int = None,
        max_tokens: int = None,
        default_target_language: str = None,
        article_timeout: int = None,
        article_max_chars: int = None,
        http_retries: int = None,
        http_backoff: float = None,
        user_agent: Optional[str] = None,
        use_head_content_type_check: bool = True,
    ):
        # ------------- Config -------------
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.temperature = self._coalesce(temperature, self.DEFAULT_TEMPERATURE)
        self.max_tokens = self._coalesce(max_tokens, self.DEFAULT_MAX_TOKENS)
        self.default_target_language = default_target_language or self.DEFAULT_TARGET_LANGUAGE
        self.article_timeout = self._coalesce(article_timeout, self.DEFAULT_HTTP_TIMEOUT)
        self.article_max_chars = self._coalesce(article_max_chars, self.DEFAULT_MAX_CHARS)
        self.http_retries = self._coalesce(http_retries, self.DEFAULT_RETRIES)
        self.http_backoff = self._coalesce(http_backoff, self.DEFAULT_BACKOFF)
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
        self.use_head_content_type_check = use_head_content_type_check

        # ------------- LLM -------------
        self.model = init_llm(
            self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Built artifacts
        self._tools = {}
        self._article_subagent = None
        self._image_subagent = None
        self._supervisor_agent = None

        # Build everything on init
        self._build_tools()
        self._build_agents()

    
    def run(self, url: str, target_language: Optional[str] = None) -> str:
        original_lang = self.default_target_language
        if target_language:
            self.default_target_language = target_language
        try:
            response = self._supervisor_agent.invoke({"messages": [{"role": "user", "content": url}]})
            ai_msg = response["messages"][-1]
            return getattr(ai_msg, "content", ai_msg)
        finally:
            self.default_target_language = original_lang
    

    # =========================================================
    # Builders
    # =========================================================
    def _build_tools(self) -> None:
        """
        Define tools as closures capturing 'self'. Decorate with @tool to make them
        LangChain tools. Store for later use by sub-agents.
        """

        # --------------- Tool 1: get_web_article ---------------
        @tool
        def get_web_article(article_url: str) -> str:
            """
            Get the article content from the website URL given by the user.
            Preserves paragraph boundaries and avoids merging into one paragraph.
            """
            text, _ = self._fetch_and_extract_article(article_url)
            # Hard limit for token safety
            return text[: self.article_max_chars]

        # --------------- Tool 2: get_translation ---------------
        
        # --- Tool 2: get_translation ---
        @tool
        def get_translation(target_language: Optional[str] = None, text: str = "") -> str:
            """
            Translate given text into `target_language`. If not provided, uses the agent's default.
            Returns ONLY the translated text (no explanations, no quotes).
            """
            lang = target_language or self.default_target_language
            system = (
                f"You are a professional translator. Precisely translate into {lang}. "
                "Return ONLY the accurate translated text. No explanations, no quotes."
            )
            user = text
            messages = [("system", system), ("user", user)]
            response = self.model.invoke(messages)
            return response.content

        # --- Tool 3: get_web_image ---
        @tool
        def get_web_image(target_language: Optional[str] = None, image_url: str = "") -> str:
            """
            Describe the image in detail in the expected target language.
            If not provided, uses the agent's default target language.
            """
            lang = target_language or self.default_target_language
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe the image in detail in {lang}."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }]
            response = self.model.invoke(messages)
            return response.content


        # --------------- Sub-agent Tools (wrappers) ---------------
        # WEB_ARTICLE_AGENT_PROMPT = (
        #     "You are a helpful Ai assistant."
        #     "Help user to get accurate information from the given website url."
        #     "First use the tool get_web_article to get the article content, do not remove any part, "
        #     "Do not do any summary or elaboration, do not merge the text into one paragraph."
        #     "Then use the tool get_translation to get the translated version of the content, "
        #     "The target language is Chinese by default, do not remove any part, do not do any summary or elaboration, do not merge the text into one paragraph."
        #     "Return ONLY the translated context, no need for the original text."
        # )

        # WEB_IMAGE_PROMPT = (
        #     "You are a helpful Ai assistant."
        #     "Help user to get accurate information from the given website url."
        #     "Use the tool get_web_image to get the image information."
        #     "Describe the image in user expected target language, the target language is Chinese by default."
        # )

        
        # --- Prompts (remove hardcoded 'Chinese') ---
        WEB_ARTICLE_AGENT_PROMPT = (
            "You are a helpful AI assistant. "
            "First use get_web_article to fetch the article content (preserve paragraphs, do not summarize). "
            "Then use get_translation to translate the content into the target language. "
            "If the target language is not provided, use the agent's default. "
            "Return ONLY the translated text."
        )

        WEB_IMAGE_PROMPT = (
            "You are a helpful AI assistant. "
            "Use get_web_image to describe the image in the expected target language. "
            "If the target language is not provided, use the agent's default."
        )


        # Pre-create sub-agents here so their wrappers can reference them
        article_subagent = create_agent(
            self.model,
            tools=[get_web_article, get_translation],
            system_prompt=WEB_ARTICLE_AGENT_PROMPT,
        )
        image_subagent = create_agent(
            self.model,
            tools=[get_web_image],
            system_prompt=WEB_IMAGE_PROMPT,
        )

        self._article_subagent = article_subagent
        self._image_subagent = image_subagent

        # Wrap sub-agents as tools so the supervisor can call exactly one
        @tool
        def web_article_event(request: str) -> str:
            """
            Web article events.
            Use this when the user gives a URL for an article.
            """
            result = article_subagent.invoke({"messages": [{"role": "user", "content": request}]})
            last = result["messages"][-1]
            return getattr(last, "content", last)

        @tool
        def web_image_event(request: str) -> str:
            """
            Web image events.
            Use this when the user gives a URL for an image.
            """
            result = image_subagent.invoke({"messages": [{"role": "user", "content": request}]})
            last = result["messages"][-1]
            return getattr(last, "content", last)

        # Keep references
        self._tools = {
            "get_web_article": get_web_article,
            "get_translation": get_translation,
            "get_web_image": get_web_image,
            "web_article_event": web_article_event,
            "web_image_event": web_image_event,
        }

    def _build_agents(self) -> None:
        """
        Build the supervisor agent that chooses between article vs. image tool.
        """
        SUPERVISOR_PROMPT = (
            "You are a helpful AI assistant.\n"
            "You can help users get detailed information when given a website URL.\n"
            "Choose exactly one tool based on whether the URL is an image or an article.\n"
            "When a tool is called, return ONLY the tool's output as the final answer as AIMessage — "
            "do not add extra commentary."
        )

        self._supervisor_agent = create_agent(
            self.model,
            tools=[self._tools["web_article_event"], self._tools["web_image_event"]],
            system_prompt=SUPERVISOR_PROMPT,
        )

    # =========================================================
    # Internals: Networking & Extraction
    # =========================================================
    def _fetch_and_extract_article(self, url: str) -> Tuple[str, str]:
        """
        Fetch HTML and extract readable paragraph text while preserving paragraph breaks.
        Returns (text, content_type).
        """
        # Optional routing: if it's clearly an image URL, return hint
        if self._looks_like_image_url(url):
            return "The provided URL appears to be an image, not an article.", "image/*"

        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"}
        content_type = None

        # Optional HEAD check for content type
        if self.use_head_content_type_check:
            try:
                head = requests.head(url, headers=headers, timeout=self.article_timeout, allow_redirects=True)
                content_type = head.headers.get("Content-Type", "")
                if content_type and "image" in content_type.lower():
                    return "The provided URL appears to be an image, not an article.", content_type
            except Exception:
                # Non-fatal; continue with GET
                pass

        html = self._http_get_with_retries(url, headers=headers, timeout=self.article_timeout)
        if html is None:
            return "Failed to fetch content from the URL.", content_type or ""

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Prefer <article>, then <main>, else <body>
        container = soup.find("article") or soup.find("main") or soup.find("body") or soup
        paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]

        # Fallback: if no <p> found, try text blocks in divs
        if not paragraphs:
            blocks = []
            for div in container.find_all("div"):
                txt = div.get_text(" ", strip=True)
                if txt and len(txt.split()) > 5:  # heuristic to skip very short fragments
                    blocks.append(txt)
            paragraphs = blocks

        # Clean and preserve paragraph breaks, avoid merging into one line
        paragraphs = [self._clean_whitespace(p) for p in paragraphs if p]
        text = self._truncate_by_chars("\n\n".join(paragraphs), self.article_max_chars)

        return text, content_type or "text/html"

    def _http_get_with_retries(self, url: str, headers: dict, timeout: int) -> Optional[str]:
        last_err = None
        for attempt in range(self.http_retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                last_err = e
                if attempt < self.http_retries:
                    time.sleep(self.http_backoff * (attempt + 1))
        return None



    # =========================================================
    # Utilities
    # =========================================================
    @staticmethod
    def _coalesce(x, default):
        return x if x is not None else default

    @staticmethod
    def _clean_whitespace(s: str) -> str:
        # Collapse internal whitespace but preserve sentence spacing
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _truncate_by_chars(s: str, max_chars: int) -> str:
        if len(s) <= max_chars:
            return s
        # Cut at nearest paragraph or sentence boundary, if possible
        cut = s[:max_chars]
        last_para = cut.rfind("\n\n")
        if last_para > max_chars * 0.7:
            return cut[:last_para]
        last_period = max(cut.rfind("."), cut.rfind("。"))
        if last_period > max_chars * 0.7:
            return cut[: last_period + 1]
        return cut

    def _looks_like_image_url(self, url: str) -> bool:
        lower = url.lower()
        return any(lower.endswith(ext) for ext in self.IMAGE_EXTENSIONS)


# =============================================================
# Example usage
# =============================================================
if __name__ == "__main__":
    agent = WebURLDetailAgent(
        model_name="gpt-4o",
        temperature=0,
        max_tokens=12_000,
        default_target_language="Japanese",
        article_timeout=15,
        article_max_chars=10_000,
        http_retries=2,
        http_backoff=0.8,
        use_head_content_type_check=True,
    )
