import argparse
import ast
import datetime
import json
import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import requests

from model_config import load_environment

load_environment()


APP_DIR = Path(__file__).resolve().parent
LOGOS_DIR = APP_DIR / "logos"
WAITING_STATUS_TEXT = "The assistant is preparing a response..."
WAITING_STATUS_HTML = (
    '<div class="status-line" role="status" aria-live="polite">'
    '<span class="status-spinner" aria-hidden="true"></span>'
    f"<span>{WAITING_STATUS_TEXT}</span>"
    "</div>"
)
SESSION_TIMEOUT_MINUTES = int(os.environ.get("MND_UI_SESSION_TIMEOUT_MINUTES", "120"))
ALLOWED_PARTICIPANT_IDS = {
    item.strip().lower()
    for item in os.environ.get("MND_ALLOWED_PARTICIPANT_IDS", "").split(",")
    if item.strip()
}
PARTICIPANT_SESSION_DIR = Path(
    os.environ.get(
        "MND_PARTICIPANT_SESSION_DIR",
        str(APP_DIR / ".runtime" / "participant_sessions"),
    )
).expanduser()
REQUIRE_PARTICIPANT_LOGIN = os.environ.get(
    "MND_REQUIRE_PARTICIPANT_LOGIN",
    "0",
).lower() not in {"0", "false", "no", "off"}
PUBLIC_DEMO_SESSION_PREFIX = os.environ.get(
    "MND_PUBLIC_DEMO_SESSION_PREFIX",
    "public-demo",
)


PRE_TEST_TOPICS = [
    "I am thinking about genetic testing for MND. What should I know first?",
    "What can a genetic test tell me, and what can it not tell me?",
    "What would a positive, negative, or uncertain result mean?",
    "How might genetic testing affect my children or other relatives?",
    "What happens during the testing process?",
    "What questions should I ask my genetics team before deciding?",
]


DEMO_CONVERSATION = [
    {
        "role": "user",
        "content": (
            "I have recently been diagnosed with MND and my neurologist mentioned genetic "
            "testing. I am not sure what it would mean for me or my family."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "I’m sorry you’re having to think about this at an already difficult time. "
            "Genetic testing for MND usually looks for changes in genes known to be linked "
            "with MND. A result may help explain why MND happened, may guide eligibility for "
            "some research or treatment options, and can sometimes give relatives clearer "
            "information about their own risk.\n\n"
            "Before going further, could you tell me a little about your immediate family, "
            "for example whether you have children, siblings, or a partner?"
        ),
    },
    {
        "role": "user",
        "content": (
            "I have two adult children and one brother. Nobody else in the family has had "
            "MND as far as I know."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Thank you, that’s helpful. Many people with MND do not have an obvious family "
            "history, and genetic testing can still sometimes find a relevant gene change. "
            "If the test found a clear inherited change, your adult children and brother "
            "might be offered their own genetics appointment to talk through whether they "
            "want predictive testing. If the test was negative or uncertain, the meaning for "
            "relatives would usually be different.\n\n"
            "Have you thought about who in your family you might want to tell about testing, "
            "or whether anyone may prefer not to know?"
        ),
    },
]


CUSTOM_CSS = """
:root {
    --mnd-ink: #24323f;
    --mnd-muted: #60707f;
    --mnd-line: #d8e3e8;
    --mnd-blue: #1f6f8b;
    --mnd-teal: #1f8a7a;
    --mnd-coral: #c75c4f;
    --mnd-soft: #f5f9fa;
    --mnd-panel: #ffffff;
}

.gradio-container {
    max-width: min(1680px, 96vw) !important;
    margin: 0 auto !important;
    color: var(--mnd-ink);
    background:
        linear-gradient(180deg, #eef7f8 0%, #f8fbfb 42%, #ffffff 100%);
}

#hero {
    padding: 28px 28px 20px;
    border-bottom: 1px solid var(--mnd-line);
}

#hero h1 {
    margin: 0 0 8px;
    font-size: 34px;
    line-height: 1.12;
    color: var(--mnd-ink);
}

#hero p {
    max-width: 820px;
    margin: 0;
    color: var(--mnd-muted);
    font-size: 16px;
    line-height: 1.55;
}

.trust-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}

.trust-pill {
    border: 1px solid var(--mnd-line);
    background: rgba(255, 255, 255, 0.72);
    border-radius: 999px;
    padding: 8px 12px;
    color: #365161;
    font-size: 13px;
    font-weight: 650;
}

#content-grid {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(320px, 380px);
    gap: 20px;
    align-items: start;
    padding: 18px;
}

#main-chat, #support-panel {
    padding: 18px;
}

#main-chat {
    min-width: 0;
}

#support-panel {
    position: sticky;
    top: 18px;
}

#chatbot {
    border: 1px solid var(--mnd-line);
    border-radius: 10px;
    overflow: hidden;
    background: var(--mnd-panel);
    box-shadow: 0 18px 50px rgba(36, 50, 63, 0.08);
}

#chatbot .wrap,
#chatbot .bubble-wrap,
#chatbot .message-wrap {
    min-height: 0 !important;
}

#chatbot button[title="Share"],
#chatbot button[title="Copy"],
#chatbot button[title="Delete"],
#chatbot button[aria-label="Share"],
#chatbot button[aria-label="Copy"],
#chatbot button[aria-label="Delete"] {
    display: none !important;
}

#message-box textarea {
    border-radius: 10px !important;
    border-color: var(--mnd-line) !important;
    font-size: 16px !important;
}

#message-input-row {
    align-items: flex-end;
    gap: 8px;
}

#voice-button {
    max-width: 54px;
    min-width: 54px;
}

#voice-button button,
button#voice-button {
    min-width: 48px !important;
    width: 48px !important;
    height: 48px !important;
    border-radius: 999px !important;
    padding: 0 !important;
    font-size: 22px !important;
    line-height: 1 !important;
}

#send-row button {
    min-height: 44px;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

#send-button {
    background: var(--mnd-blue) !important;
    border-color: var(--mnd-blue) !important;
    color: white !important;
}

button[title="Stop"],
button[aria-label="Stop"],
button[title="Cancel"],
button[aria-label="Cancel"],
button.stop,
.stop-button,
.mnd-hide-generated-control {
    display: none !important;
}

#clear-button {
    color: var(--mnd-coral) !important;
}

#login-page {
    max-width: 760px;
    margin: 34px auto;
    padding: 26px;
    border: 1px solid var(--mnd-line);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 18px 50px rgba(36, 50, 63, 0.08);
}

#login-page h2 {
    margin-top: 0;
    color: var(--mnd-ink);
}

#app-page,
#app-page > .block,
#app-page > .form,
#app-page > .gr-group,
#app-page > div,
#app-page > div > .block,
#app-page > div > .form,
#app-page > div > .gr-group {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    outline: 0 !important;
}

#app-page {
    padding: 0 !important;
    margin: 0 !important;
}

#app-page > div,
#app-page > div > .block,
#app-page > div > .form,
#app-page > div > .gr-group {
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

#app-page [style*="display: none"],
#app-page .hide,
#app-page .hidden {
    display: none !important;
}

body.mnd-logged-in #login-page {
    display: none !important;
}

#login-page:has(.hidden),
#login-page:has(button.hidden),
#login-page:has(.form.hidden) {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    overflow: hidden !important;
}

body.mnd-logged-in #app-page,
body.mnd-logged-in #app-page > .block,
body.mnd-logged-in #app-page > .form,
body.mnd-logged-in #app-page > .gr-group,
body.mnd-logged-in #app-page > div,
body.mnd-logged-in #app-page > div > .block,
body.mnd-logged-in #app-page > div > .form,
body.mnd-logged-in #app-page > div > .gr-group {
    display: contents !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

#participant-status {
    color: var(--mnd-muted);
    font-weight: 700;
}

#participant-login-row {
    align-items: flex-end;
}

#participant-prefix {
    max-width: 76px;
    margin-bottom: 8px;
    font-size: 18px;
    font-weight: 800;
    color: var(--mnd-ink);
}

#voice-status {
    color: var(--mnd-muted);
    font-size: 14px;
    font-weight: 650;
}

.info-panel {
    border: 1px solid var(--mnd-line);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.78);
    padding: 16px;
    margin-bottom: 14px;
}

.info-panel h2, .info-panel h3 {
    margin: 0 0 8px;
    color: var(--mnd-ink);
}

.info-panel p, .info-panel li {
    color: var(--mnd-muted);
    line-height: 1.48;
}

.info-panel ul {
    padding-left: 18px;
    margin-bottom: 0;
}

.starter-button button {
    justify-content: flex-start !important;
    white-space: normal !important;
    text-align: left !important;
    min-height: 42px;
    border-radius: 8px !important;
    border-color: var(--mnd-line) !important;
    background: #ffffff !important;
    color: var(--mnd-ink) !important;
    font-weight: 600 !important;
}

.starter-button button:hover {
    border-color: var(--mnd-teal) !important;
    box-shadow: 0 8px 24px rgba(31, 138, 122, 0.12);
}

#logos {
    align-items: center;
}

#logos img {
    object-fit: contain !important;
}

.accessibility-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    padding: 14px 28px 0;
}

.accessibility-note {
    color: var(--mnd-muted);
    font-size: 14px;
    font-weight: 650;
    margin-right: 4px;
}

#accessibility-row button, #download-button {
    min-height: 40px;
    border-radius: 8px !important;
    font-weight: 740 !important;
}

#read-button {
    background: #ffffff !important;
    border-color: var(--mnd-teal) !important;
    color: #146f65 !important;
}

#download-button {
    color: #365161 !important;
}

.prototype-reminder {
    margin: 14px 0 10px;
    padding: 11px 14px;
    border: 1px solid var(--mnd-line);
    border-left: 4px solid var(--mnd-teal);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.82);
    color: var(--mnd-muted);
    font-size: 14px;
    line-height: 1.42;
}

#session-timeout {
    display: none;
    margin: 10px 0;
    padding: 12px 14px;
    border: 1px solid #f2c94c;
    border-left: 5px solid #b7791f;
    border-radius: 8px;
    background: #fff8df;
    color: #47380b;
    font-weight: 700;
}

#response-status {
    margin: 10px 0 6px;
    padding: 12px 14px;
    border: 1px solid #b9d8e6;
    border-left: 5px solid var(--mnd-blue);
    border-radius: 8px;
    background: #eaf6fb;
    color: var(--mnd-ink);
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(31, 111, 139, 0.10);
}

#response-status p {
    margin: 0 !important;
}

#response-status .status-line {
    display: flex;
    align-items: center;
    gap: 10px;
}

#response-status .status-spinner {
    width: 18px;
    height: 18px;
    flex: 0 0 18px;
    border: 3px solid rgba(31, 111, 139, 0.24);
    border-top-color: var(--mnd-blue);
    border-radius: 999px;
    animation: mnd-status-spin 0.9s linear infinite;
}

@keyframes mnd-status-spin {
    to {
        transform: rotate(360deg);
    }
}

/* Hide Gradio queue/progress elapsed-time labels in the patient-facing demo. */
.progress-text,
.progress-level,
[data-testid="block-info"],
[class*="progress"] [class*="eta"],
[class*="progress"] [class*="time"] {
    display: none !important;
}

html.mnd-large-text .gradio-container,
body.mnd-large-text .gradio-container {
    font-size: 132% !important;
}

html.mnd-large-text #hero h1,
body.mnd-large-text #hero h1 {
    font-size: 44px;
}

html.mnd-large-text #chatbot,
body.mnd-large-text #chatbot {
    font-size: 20px !important;
}

html.mnd-large-text #chatbot .message,
html.mnd-large-text #chatbot .message *,
html.mnd-large-text #chatbot .prose,
html.mnd-large-text #chatbot .prose *,
html.mnd-large-text #chatbot p,
html.mnd-large-text #chatbot li,
body.mnd-large-text #chatbot .message,
body.mnd-large-text #chatbot .message *,
body.mnd-large-text #chatbot .prose,
body.mnd-large-text #chatbot .prose *,
body.mnd-large-text #chatbot p,
body.mnd-large-text #chatbot li {
    font-size: 20px !important;
    line-height: 1.62 !important;
}

html.mnd-large-text #message-box textarea,
body.mnd-large-text #message-box textarea {
    font-size: 22px !important;
}

html.mnd-large-text .info-panel p,
html.mnd-large-text .info-panel li,
body.mnd-large-text .info-panel p,
body.mnd-large-text .info-panel li {
    font-size: 18px !important;
}

html.mnd-high-contrast .gradio-container,
body.mnd-high-contrast .gradio-container {
    background: #ffffff !important;
    color: #101820 !important;
    --body-text-color: #101820;
    --body-text-color-subdued: #263442;
    --border-color-primary: #101820;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f2f2f2;
}

html.mnd-high-contrast #chatbot,
html.mnd-high-contrast .info-panel,
html.mnd-high-contrast #message-box textarea,
body.mnd-high-contrast #chatbot,
body.mnd-high-contrast .info-panel,
body.mnd-high-contrast #message-box textarea {
    border-color: #111827 !important;
    box-shadow: none !important;
}

html.mnd-high-contrast #hero,
html.mnd-high-contrast #hero *,
html.mnd-high-contrast .info-panel,
html.mnd-high-contrast .info-panel *,
html.mnd-high-contrast .prototype-reminder,
html.mnd-high-contrast #response-status,
body.mnd-high-contrast #hero,
body.mnd-high-contrast #hero *,
body.mnd-high-contrast .info-panel,
body.mnd-high-contrast .info-panel *,
body.mnd-high-contrast .prototype-reminder {
    color: #101820 !important;
    background-color: #ffffff !important;
}

body.mnd-high-contrast #response-status {
    color: #101820 !important;
    background-color: #ffffff !important;
    border-color: #101820 !important;
}

html.mnd-high-contrast #response-status .status-spinner,
body.mnd-high-contrast #response-status .status-spinner {
    border-color: #d1d5db !important;
    border-top-color: #101820 !important;
}

html.mnd-high-contrast #chatbot .message,
html.mnd-high-contrast #chatbot .message.bot,
html.mnd-high-contrast #chatbot .message-row.bot .message,
body.mnd-high-contrast #chatbot .message,
body.mnd-high-contrast #chatbot .message.bot,
body.mnd-high-contrast #chatbot .message-row.bot .message {
    color: #101820 !important;
    background: #ffffff !important;
    border: 2px solid #101820 !important;
}

html.mnd-high-contrast #chatbot .message.user,
html.mnd-high-contrast #chatbot .message-row.user .message,
body.mnd-high-contrast #chatbot .message.user,
body.mnd-high-contrast #chatbot .message-row.user .message {
    color: #ffffff !important;
    background: #101820 !important;
    border: 2px solid #101820 !important;
}

html.mnd-high-contrast #chatbot .message *,
html.mnd-high-contrast #chatbot .prose *,
body.mnd-high-contrast #chatbot .message *,
body.mnd-high-contrast #chatbot .prose * {
    color: inherit !important;
}

html.mnd-high-contrast #chatbot a,
body.mnd-high-contrast #chatbot a {
    color: #0047b3 !important;
    text-decoration: underline !important;
    font-weight: 700 !important;
}

html.mnd-high-contrast #chatbot .message.user a,
html.mnd-high-contrast #chatbot .message-row.user .message a,
body.mnd-high-contrast #chatbot .message.user a,
body.mnd-high-contrast #chatbot .message-row.user .message a {
    color: #ffff00 !important;
}

html.mnd-high-contrast button:focus-visible,
html.mnd-high-contrast textarea:focus-visible,
body.mnd-high-contrast button:focus-visible,
body.mnd-high-contrast textarea:focus-visible {
    outline: 4px solid #ffbf00 !important;
    outline-offset: 2px !important;
}

html.mnd-high-contrast .trust-pill,
html.mnd-high-contrast .starter-button button,
body.mnd-high-contrast .trust-pill,
body.mnd-high-contrast .starter-button button {
    border-color: #111827 !important;
    color: #101820 !important;
    background: #ffffff !important;
}

@media (max-width: 760px) {
    #hero {
        padding: 22px 16px 14px;
    }

    #hero h1 {
        font-size: 26px;
    }

    #content-grid {
        display: block !important;
        padding: 10px;
    }

    #main-chat, #support-panel {
        padding: 10px;
    }

    #support-panel {
        position: static;
    }
}

@media (max-width: 1080px) {
    #content-grid {
        grid-template-columns: 1fr;
    }

    #support-panel {
        position: static;
    }
}
"""

ENTER_TO_SEND_JS = """
() => {
    if (window.__mndEnterToSendBound) return;
    window.__mndEnterToSendBound = true;
    const removeGeneratedControls = () => {
        document.querySelectorAll("button").forEach((button) => {
            const label = [
                button.getAttribute("aria-label"),
                button.getAttribute("title"),
                button.textContent,
            ].filter(Boolean).join(" ").trim().toLowerCase();
            const squareOnly = ["■", "□", "▣", "▢", "◼", "◻", "stop"].includes(
                button.textContent.trim().toLowerCase()
            );
            if (
                label === "stop" ||
                label === "cancel" ||
                label.includes("stop generating") ||
                label.includes("cancel") ||
                squareOnly
            ) {
                button.classList.add("mnd-hide-generated-control");
                button.setAttribute("aria-hidden", "true");
                button.setAttribute("tabindex", "-1");
            }
        });
    };
    removeGeneratedControls();
    const observer = new MutationObserver(removeGeneratedControls);
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    document.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" || event.isComposing) {
            return;
        }
        const path = typeof event.composedPath === "function"
            ? event.composedPath()
            : [event.target];
        const textarea = path.find((node) => node instanceof HTMLTextAreaElement);
        const isMessageBox = path.some((node) => node && node.id === "message-box")
            || (textarea && textarea.closest && textarea.closest("#message-box"));
        if (!textarea || !isMessageBox) return;

        if (event.shiftKey) {
            event.stopImmediatePropagation();
            return;
        }

        const sendButton = document.querySelector(
            "button#send-button, #send-button button, #send-button"
        );
        if (!sendButton || sendButton.disabled) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        sendButton.click();
    }, true);
}
"""

SESSION_TIMEOUT_JS = f"""
() => {{
    if (window.__mndSessionTimeoutBound) return;
    window.__mndSessionTimeoutBound = true;
    const timeoutMs = {SESSION_TIMEOUT_MINUTES} * 60 * 1000;
    let timer = null;

    const disableDemo = () => {{
        const notice = document.querySelector("#session-timeout");
        if (notice) {{
            notice.style.display = "block";
        }}
        document.querySelectorAll(
            "#message-box textarea, #send-button button, button#send-button"
        ).forEach((element) => {{
            element.disabled = true;
            element.setAttribute("aria-disabled", "true");
        }});
    }};

    const resetTimer = () => {{
        if (timer) clearTimeout(timer);
        timer = setTimeout(disableDemo, timeoutMs);
    }};

    ["click", "keydown", "mousemove", "touchstart", "scroll"].forEach((eventName) => {{
        document.addEventListener(eventName, resetTimer, true);
    }});
    resetTimer();
}}
"""

VOICE_INPUT_JS = """
() => {
    if (window.__mndVoiceInputBound) return;
    window.__mndVoiceInputBound = true;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    const getButton = () => document.querySelector(
        "button#voice-button, #voice-button button, #voice-button"
    );
    const getTextarea = () => document.querySelector("#message-box textarea");
    const getStatus = () => document.querySelector("#voice-status");

    const setStatus = (message) => {
        const status = getStatus();
        if (!status) return;
        status.textContent = message || "";
    };

    const setButton = (label, title, listening) => {
        const button = getButton();
        if (!button) return;
        button.textContent = label;
        button.title = title;
        button.setAttribute("aria-label", title);
        button.setAttribute("aria-pressed", listening ? "true" : "false");
    };

    const setTextareaValue = (value) => {
        const textarea = getTextarea();
        if (!textarea) return;
        const setter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype,
            "value"
        ).set;
        setter.call(textarea, value);
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.dispatchEvent(new Event("change", { bubbles: true }));
        textarea.focus();
    };

    const startDictation = () => {
        if (!SpeechRecognition) {
            setStatus(
                "Voice input is not supported in this browser. Please use Chrome or Edge, or type your message."
            );
            return;
        }

        const textarea = getTextarea();
        if (!textarea) {
            setStatus("Message box is not ready yet. Please try again.");
            return;
        }

        const initialText = textarea.value.trim();
        const recognition = new SpeechRecognition();
        let settled = false;
        let heardResult = false;
        let lastError = "";
        let finalTranscript = "";

        recognition.lang = navigator.language || "en-US";
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;

        const finish = (status) => {
            if (settled) return;
            settled = true;
            setButton("🎙", "Dictate message", false);
            setStatus(status || "");
        };

        recognition.onstart = () => {
            setButton("●", "Listening", true);
            setStatus("Listening...");
        };

        recognition.onerror = (event) => {
            lastError = event && event.error ? event.error : "";
            if (lastError === "aborted" || lastError === "no-speech") {
                return;
            }

            const permissionErrors = new Set(["not-allowed", "service-not-allowed"]);
            const message = permissionErrors.has(lastError)
                ? "Microphone permission was blocked. Please allow microphone access in your browser, or type instead."
                : `Voice input stopped: ${lastError || "unknown error"}. You can type instead.`;
            finish(message);
        };

        recognition.onend = () => {
            if (finalTranscript.trim()) {
                finish("Voice input added to the message box. Please review it before sending.");
                return;
            }
            if (!heardResult && !settled) {
                const message = lastError === "no-speech"
                    ? "I did not catch any speech. Please try again or type instead."
                    : "";
                finish(message);
            }
        };

        recognition.onresult = (event) => {
            heardResult = true;
            const chunks = [];
            for (let i = event.resultIndex; i < event.results.length; i += 1) {
                const result = event.results[i];
                const transcript = result[0] && result[0].transcript
                    ? result[0].transcript
                    : "";
                if (result.isFinal) {
                    finalTranscript += ` ${transcript}`;
                }
                chunks.push(transcript);
            }

            const spokenText = (finalTranscript || chunks.join(" ")).trim();
            if (!spokenText) return;

            const combined = initialText && spokenText
                ? `${initialText} ${spokenText}`.trim()
                : spokenText;
            setTextareaValue(combined);
            setStatus("Listening... text is appearing in the message box.");
        };

        try {
            recognition.start();
        } catch (error) {
            finish("Voice input could not start. Please check microphone permission or type instead.");
        }
    };

    document.addEventListener("click", (event) => {
        const target = event.target;
        const voiceButton = target && target.closest
            ? target.closest("button#voice-button, #voice-button button, #voice-button")
            : null;
        if (!voiceButton) return;
        event.preventDefault();
        event.stopPropagation();
        startDictation();
    }, true);
}
"""

LOGIN_SUCCESS_JS = """
(participantId) => {
    if (!participantId) return;
    document.body.classList.add("mnd-logged-in");
    document.querySelectorAll("#login-page").forEach((element) => {
        element.remove();
    });
}
"""


GREETINGS = [
    (
        "Hi, I’m glad we can talk today. My role is to support you in thinking through "
        "genetic testing for MND, and to make sure you have the information you need to "
        "decide whether it feels right for you.\n\n"
        "Before we talk about the test itself, would it be okay if I ask a few questions "
        "about your situation and your family? This helps me tailor the information to you. "
        "You can skip any of these questions and ask for general information instead."
    ),
    (
        "Hello, and thank you for talking with me today. I’m here to help you think through "
        "genetic testing for MND and to explain what it can and can’t tell you so that you "
        "can decide whether it feels right for you.\n\n"
        "To start, would it be alright if I ask a few questions about you and your family? "
        "It helps me make the information as relevant as possible. You can skip any question "
        "and ask for general information instead."
    ),
    (
        "Hi, it’s good to meet you. My role is to support you as you consider genetic testing "
        "for MND and to help make sure you have clear information before you decide what to do.\n\n"
        "Before we get into the details of the test, would it be okay if I ask a few questions "
        "about your situation and your family, so I can tailor what I say to you? You can "
        "skip any question and ask for general information instead."
    ),
]


def build_openai_messages(message):
    """
    The backend owns the counselling system prompt, stage control, and
    conversation state through MNDAgent.ask(). The frontend sends only the
    latest user turn to the OpenAI-compatible local endpoint.
    """
    return [{"role": "user", "content": message}]


def stream_openai_compatible_chat(
    message,
    history,
    api_base,
    model,
    api_key,
    temperature,
    max_tokens,
    greeting,
    session_id,
):
    payload = {
        "model": model,
        "messages": build_openai_messages(message),
        "stream": True,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)

    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    url = f"{base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Session-ID": session_id,
    }

    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    answer = ""

    try:
        with requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=300,
            stream=True,
        ) as response:
            response.raise_for_status()

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()

                if line == "[DONE]":
                    break

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                chunk = delta.get("content")

                if chunk is None:
                    message_obj = choice.get("message", {})
                    chunk = message_obj.get("content")

                if chunk:
                    answer += chunk
                    yield answer

    except Exception as exc:
        yield (
            "Error calling the agent API.\n\n"
            f"Details: {exc}\n\n"
            "Please check that the backend is running and that the API base is correct."
        )


def normalise_participant_id(participant_id):
    return str(participant_id or "").strip().lower()


def compose_participant_id(participant_number, access_code=""):
    participant_number = normalise_participant_id(participant_number)
    access_code = normalise_participant_id(access_code)

    if participant_number.startswith("aigc-"):
        return participant_number

    participant_number = participant_number.removeprefix("aigc-").strip("- ")
    access_code = access_code.strip("- ")
    if not participant_number or not access_code:
        return ""

    return f"aigc-{participant_number}-{access_code}"


def split_participant_id(participant_id):
    participant_id = normalise_participant_id(participant_id)
    if not participant_id.startswith("aigc-"):
        return "", ""

    remainder = participant_id[len("aigc-") :]
    parts = remainder.split("-", 1)
    if len(parts) != 2:
        return remainder, ""
    return parts[0], parts[1]


def message_content_to_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return value
            parsed_text = message_content_to_text(parsed)
            return parsed_text or value
        return value

    if isinstance(value, list):
        parts = [message_content_to_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()

    if isinstance(value, dict):
        if "text" in value:
            return message_content_to_text(value.get("text"))
        if "content" in value:
            return message_content_to_text(value.get("content"))
        if "value" in value:
            return message_content_to_text(value.get("value"))
        return ""

    return str(value)


def participant_transcript_path(participant_id):
    participant_id = normalise_participant_id(participant_id)
    if participant_id not in ALLOWED_PARTICIPANT_IDS:
        raise ValueError("Invalid participant ID.")
    return PARTICIPANT_SESSION_DIR / f"{participant_id}.json"


def load_participant_history(participant_id, greeting):
    participant_id = normalise_participant_id(participant_id)
    if participant_id not in ALLOWED_PARTICIPANT_IDS:
        return []

    path = participant_transcript_path(participant_id)
    if not path.exists():
        return [{"role": "assistant", "content": greeting}]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [{"role": "assistant", "content": greeting}]

    history = payload.get("history", [])
    if not isinstance(history, list):
        return [{"role": "assistant", "content": greeting}]

    cleaned_history = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = message_content_to_text(message.get("content", ""))
        if role in {"user", "assistant"} and content:
            cleaned_history.append({"role": role, "content": content})

    return cleaned_history or [{"role": "assistant", "content": greeting}]


def save_participant_history(history, participant_id):
    participant_id = normalise_participant_id(participant_id)
    if participant_id not in ALLOWED_PARTICIPANT_IDS:
        return None

    cleaned_history = []
    for message in history or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = message_content_to_text(message.get("content", ""))
        if role in {"user", "assistant"} and content:
            cleaned_history.append({"role": role, "content": content})

    try:
        PARTICIPANT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path = participant_transcript_path(participant_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "participant_id": participant_id,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "history": cleaned_history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        return None

    return None


def reset_backend_session(api_base, session_id):
    if not session_id:
        return

    url = f"{api_base.rstrip('/').removesuffix('/v1')}/reset"
    try:
        requests.post(
            url,
            json={"session_id": session_id},
            headers={"X-Session-ID": session_id},
            timeout=10,
        )
    except requests.RequestException:
        pass


def authenticate_participant(participant_number, access_code, api_base, greeting, current_history):
    participant_id = compose_participant_id(participant_number, access_code)
    participant_number, access_code = split_participant_id(participant_id)
    disabled = gr.update(interactive=False)
    disabled_starters = [disabled] * len(PRE_TEST_TOPICS)
    enabled_starters = [gr.update(interactive=True)] * len(PRE_TEST_TOPICS)

    if participant_id not in ALLOWED_PARTICIPANT_IDS:
        return (
            "",
            "",
            current_history or [],
            gr.update(
                value=(
                    "Participant ID not recognised. Please enter the number and "
                    "access code supplied by the study team."
                ),
                visible=True,
            ),
            gr.update(value=participant_number, interactive=True),
            gr.update(value=access_code, interactive=True),
            gr.update(interactive=True),
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            gr.update(visible=REQUIRE_PARTICIPANT_LOGIN),
            gr.update(visible=not REQUIRE_PARTICIPANT_LOGIN),
            *disabled_starters,
        )

    saved_history = load_participant_history(participant_id, greeting)
    return (
        participant_id,
        participant_id,
        saved_history,
        gr.update(value=f"Signed in as `{participant_id}`.", visible=True),
        gr.update(value=participant_number, interactive=False),
        gr.update(value=access_code, interactive=False),
        disabled,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(visible=True),
        *enabled_starters,
    )


def user(user_message, history, participant_id):
    if history is None:
        history = []

    if (
        REQUIRE_PARTICIPANT_LOGIN
        and normalise_participant_id(participant_id) not in ALLOWED_PARTICIPANT_IDS
    ):
        return (
            "",
            history,
            gr.update(
                value="Please enter your participant ID before starting.",
                visible=True,
            ),
        )

    if not user_message.strip():
        return "", history, gr.update(value="", visible=False)

    return (
        "",
        history + [{"role": "user", "content": user_message}],
        gr.update(value=WAITING_STATUS_HTML, visible=True),
    )


def bot(
    history,
    api_base,
    model,
    api_key,
    temperature,
    max_tokens,
    greeting,
    session_id,
):
    if history is None or not history:
        yield history, gr.update(value="", visible=False)
        return

    last_message = history[-1]
    if not isinstance(last_message, dict) or last_message.get("role") != "user":
        yield history, gr.update(value="", visible=False)
        return

    user_message = last_message.get("content", "")
    history_for_api = history[:-1]

    history = history + [{"role": "assistant", "content": ""}]
    yield history, gr.update(value=WAITING_STATUS_HTML, visible=True)

    for answer in stream_openai_compatible_chat(
        message=user_message,
        history=history_for_api,
        api_base=api_base,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        greeting=greeting,
        session_id=session_id,
    ):
        history[-1]["content"] = answer
        yield history, gr.update(value=WAITING_STATUS_HTML, visible=True)

    yield history, gr.update(value="", visible=False)


def clear_chat(greeting):
    return [{"role": "assistant", "content": greeting}]


def reset_backend_chat(api_base, greeting, session_id):
    session_id = normalise_participant_id(session_id)
    reset_backend_session(api_base, session_id)
    history = [{"role": "assistant", "content": greeting}]
    save_participant_history(history, session_id)
    return history, session_id


# def build_initial_history(greeting):
#     return [{"role": "assistant", "content": greeting}] + DEMO_CONVERSATION
def load_prefilled_session(greeting):
    prefill_path = os.environ.get("MND_PREFILL_SESSION", "").strip()
    if not prefill_path:
        return [{"role": "assistant", "content": greeting}], None

    payload = json.loads(Path(prefill_path).read_text(encoding="utf-8"))
    history = payload.get("history")
    session_id = payload.get("session_id")
    if not isinstance(history, list) or not session_id:
        raise ValueError(
            "MND_PREFILL_SESSION must contain a history list and session_id."
        )
    if not history or history[0].get("role") != "assistant":
        history = [{"role": "assistant", "content": greeting}] + history
    return history, str(session_id)


def build_initial_history(greeting):
    history, _ = load_prefilled_session(greeting)
    return history


def export_transcript(history):
    lines = []
    for message in history or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).upper()
        content = message_content_to_text(message.get("content", "")).strip()
        if role and content:
            lines.append(f"{role}: {content}\n")

    path = os.path.join(
        tempfile.gettempdir(),
        f"mnd_conversation_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt",
    )
    with open(path, "w", encoding="utf-8") as transcript:
        transcript.write("\n".join(lines))
    return path


def build_ui(default_api_base, default_model):
    if REQUIRE_PARTICIPANT_LOGIN and not ALLOWED_PARTICIPANT_IDS:
        raise ValueError(
            "Participant login is enabled, but no participant IDs are configured. "
            "Set MND_ALLOWED_PARTICIPANT_IDS in .env or disable login with "
            "MND_REQUIRE_PARTICIPANT_LOGIN=0."
        )
    greeting = GREETINGS[0]
    initial_history, prefilled_session_id = load_prefilled_session(greeting)
    login_required = REQUIRE_PARTICIPANT_LOGIN and not prefilled_session_id
    is_prefilled_session = bool(prefilled_session_id)
    controls_enabled = not login_required
    initial_participant_id = normalise_participant_id(prefilled_session_id)
    initial_participant_number, initial_access_code = split_participant_id(
        initial_participant_id
    )
    initial_chat_history = initial_history if controls_enabled else []
    show_dev_settings = os.environ.get("MND_SHOW_DEV_SETTINGS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    with gr.Blocks(
        title="Pre-test MND Genetics Counselling",
    ) as demo:
        participant_state = gr.State(value=initial_participant_id)
        session_id = gr.State(
            value=lambda: initial_participant_id
            or f"{PUBLIC_DEMO_SESSION_PREFIX}-{uuid.uuid4().hex}"
        )
        gr.Markdown(
            """
            # MND Genetics Counselling

            A calm, private space to think through genetic testing before a clinic appointment.
            The assistant can explain what testing may mean, help prepare questions, and support
            family conversations. It does not replace advice from your own neurology or genetics team.

            <div class="trust-strip">
                <span class="trust-pill">Pre-test discussion</span>
                <span class="trust-pill">Plain-language answers</span>
                <span class="trust-pill">Family implications</span>
                <span class="trust-pill">Local research prototype</span>
            </div>
            """,
            elem_id="hero",
        )

        with gr.Group(elem_id="login-page", visible=login_required) as login_page:
            gr.Markdown(
                """
                ## Participant login

                Please enter the participant ID you were given to start the demo.
                Your conversation will be linked to this ID so you can return to it later.
                """
            )
            with gr.Row(elem_id="participant-login-row"):
                gr.Markdown("`aigc-`", elem_id="participant-prefix")
                participant_number_box = gr.Textbox(
                    label="Participant number",
                    placeholder="0xx",
                    value=initial_participant_number,
                    interactive=login_required,
                    scale=1,
                )
                access_code_box = gr.Textbox(
                    label="Access code",
                    placeholder="xxxx",
                    value=initial_access_code,
                    type="password",
                    interactive=login_required,
                    scale=1,
                )
            participant_login = gr.Button(
                "Start session",
                variant="primary",
                interactive=login_required,
            )
            participant_status = gr.Markdown(
                value=(
                    f"Signed in as `{initial_participant_id}`."
                    if is_prefilled_session
                    else "Enter your participant ID to continue."
                ),
                visible=True,
                elem_id="participant-status",
            )

        with gr.Group(elem_id="app-page", visible=controls_enabled) as app_page:
            with gr.Row(elem_id="accessibility-row", elem_classes=["accessibility-bar"]):
                gr.Markdown(
                    '<span class="accessibility-note">Accessibility</span>',
                    show_label=False,
                )
                text_size_toggle = gr.Button(
                    "Larger text",
                    size="sm",
                    elem_id="large-text-button",
                )
                contrast_toggle = gr.Button(
                    "High contrast",
                    size="sm",
                    elem_id="contrast-button",
                )
                read_aloud = gr.Button("Read latest reply", size="sm", elem_id="read-button")

            with gr.Row(elem_id="content-grid"):

                with gr.Column(elem_id="main-chat", scale=4):
                    chatbot = gr.Chatbot(
                        value=initial_chat_history,
                        height=520,
                        label="Counselling conversation",
                        elem_id="chatbot",
                    )
                    response_status = gr.HTML(
                        value="",
                        visible=False,
                        elem_id="response-status",
                    )
                    session_timeout_notice = gr.HTML(
                        value=(
                            '<div role="status">This demo session has been inactive for '
                            f"{SESSION_TIMEOUT_MINUTES} minutes. Please refresh the page "
                            "to start again, or close this tab if you are finished.</div>"
                        ),
                        visible=True,
                        elem_id="session-timeout",
                    )

                    gr.Markdown(
                        """
                        <div class="prototype-reminder">
                            This is a research prototype, not a clinician. Your neurology or genetics team should guide decisions about testing.
                        </div>
                        """
                    )

                    with gr.Row(elem_id="message-input-row"):
                        msg = gr.Textbox(
                            label="Your message",
                            placeholder="Type what you are wondering about, for example: I have MND and I am unsure whether genetic testing is right for me.",
                            lines=3,
                            max_lines=7,
                            elem_id="message-box",
                            interactive=controls_enabled,
                            scale=12,
                        )
                        voice = gr.Button(
                            "🎙",
                            elem_id="voice-button",
                            interactive=controls_enabled,
                            scale=0,
                        )
                    gr.Markdown("Press **Enter** to send.")
                    with gr.Row(elem_id="send-row"):
                        send = gr.Button(
                            "Send message",
                            variant="primary",
                            elem_id="send-button",
                            interactive=controls_enabled,
                        )
                        clear = gr.Button(
                            "Start again",
                            elem_id="clear-button",
                            interactive=controls_enabled,
                        )
                        download = gr.DownloadButton(
                            "Download conversation",
                            elem_id="download-button",
                            interactive=controls_enabled,
                        )
                    voice_status = gr.Markdown(
                        value=(
                            "Voice input uses your browser's speech recognition. "
                            "Please review the text before sending."
                        ),
                        elem_id="voice-status",
                    )

                with gr.Column(elem_id="support-panel", scale=1):
                    with gr.Accordion("Help and support", open=True):
                        gr.Markdown(
                            """
                            <div class="info-panel">
                                <h2>What this can help with</h2>
                                <ul>
                                    <li>Whether genetic testing feels right for you</li>
                                    <li>Possible test results and uncertainty</li>
                                    <li>What results may mean for relatives</li>
                                    <li>Questions to take to a genetics clinic</li>
                                </ul>
                            </div>
                            """
                        )

                        gr.Markdown(
                            """
                            <div class="info-panel">
                                <h3>Before you start</h3>
                                <p>You can share as little or as much as you want. The assistant will usually ask one question at a time so the discussion stays manageable.</p>
                            </div>
                            """
                        )

                        gr.Markdown(
                            """
                            <div class="info-panel">
                                <h3>If you would like to talk to someone</h3>
                                <p>This is a research prototype, not a clinician.</p>
                                <ul>
                                    <li>Your genetics or neurology team</li>
                                    <li>MND Association helpline: 0808 802 6262</li>
                                    <li>Samaritans: 116 123, free, any time</li>
                                </ul>
                            </div>
                            """
                        )

                    with gr.Accordion("Conversation starters", open=False):
                        starter_buttons = []
                        for topic in PRE_TEST_TOPICS:
                            starter_buttons.append(
                                gr.Button(
                                    topic,
                                    elem_classes=["starter-button"],
                                    interactive=controls_enabled,
                                )
                            )

                    if show_dev_settings:
                        settings_container = gr.Accordion("Developer settings", open=False)
                    else:
                        settings_container = gr.Group(visible=False)

                    with settings_container:
                        gr.Markdown(
                            "These settings connect this page to the counselling service. "
                            "To choose its underlying model, update `MND_AGENT_MODEL` "
                            "and `MND_AGENT_API_BASE` in `.env`, then restart the service."
                        )
                        api_base = gr.Textbox(value=default_api_base, label="Counselling service URL")
                        model = gr.Textbox(value=default_model, label="Service model alias")
                        api_key = gr.Textbox(
                            value="",
                            label="API key",
                            type="password",
                            placeholder="Optional key for the counselling service",
                        )
                        # The service owns model-specific sampling settings.
                        temperature = gr.State(value=None)
                        if show_dev_settings:
                            max_tokens = gr.Number(
                                minimum=128,
                                maximum=4096,
                                value=None,
                                precision=0,
                                label="Reply token limit (optional)",
                                info="Leave blank to use the service's configured limit.",
                            )
                        else:
                            max_tokens = gr.State(value=None)
                        greeting_box = gr.Textbox(
                            value=greeting,
                            label="Initial greeting",
                            visible=False,
                        )

                    available_logos = [
                        path
                        for name in ("mnda.png", "kings.png", "ukri.png")
                        if (path := LOGOS_DIR / name).is_file()
                    ]
                    if available_logos:
                        with gr.Accordion("Funders and partners", open=False):
                            gr.Markdown("Supported by")
                            with gr.Row(elem_id="logos"):
                                for path in available_logos:
                                    gr.Image(value=str(path), show_label=False, interactive=False, height=62, buttons=[])

        user_inputs = [msg, chatbot, participant_state]
        user_outputs = [msg, chatbot, response_status]
        bot_inputs = [
            chatbot,
            api_base,
            model,
            api_key,
            temperature,
            max_tokens,
            greeting_box,
            session_id,
        ]

        demo.load(None, None, None, js=ENTER_TO_SEND_JS, queue=False)
        demo.load(None, None, None, js=SESSION_TIMEOUT_JS, queue=False)
        demo.load(None, None, None, js=VOICE_INPUT_JS, queue=False)

        login_outputs = [
            participant_state,
            session_id,
            chatbot,
            participant_status,
            participant_number_box,
            access_code_box,
            participant_login,
            msg,
            voice,
            send,
            clear,
            download,
            login_page,
            app_page,
            *starter_buttons,
        ]

        participant_login.click(
            authenticate_participant,
            inputs=[
                participant_number_box,
                access_code_box,
                api_base,
                greeting_box,
                chatbot,
            ],
            outputs=login_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            None,
            participant_state,
            None,
            js=LOGIN_SUCCESS_JS,
            queue=False,
        )

        participant_number_box.submit(
            authenticate_participant,
            inputs=[
                participant_number_box,
                access_code_box,
                api_base,
                greeting_box,
                chatbot,
            ],
            outputs=login_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            None,
            participant_state,
            None,
            js=LOGIN_SUCCESS_JS,
            queue=False,
        )

        access_code_box.submit(
            authenticate_participant,
            inputs=[
                participant_number_box,
                access_code_box,
                api_base,
                greeting_box,
                chatbot,
            ],
            outputs=login_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            None,
            participant_state,
            None,
            js=LOGIN_SUCCESS_JS,
            queue=False,
        )

        text_size_toggle.click(
            None,
            None,
            None,
            js="""
            () => {
                const enabled = document.documentElement.classList.toggle("mnd-large-text");
                document.body.classList.toggle("mnd-large-text", enabled);
                const button = document.querySelector(
                    "button#large-text-button, #large-text-button button, #large-text-button"
                );
                if (button) {
                    button.setAttribute("aria-pressed", String(enabled));
                    button.textContent = enabled ? "Standard text" : "Larger text";
                }
            }
            """,
            queue=False,
        )

        contrast_toggle.click(
            None,
            None,
            None,
            js="""
            () => {
                const enabled = document.documentElement.classList.toggle("mnd-high-contrast");
                document.body.classList.toggle("mnd-high-contrast", enabled);
                const button = document.querySelector(
                    "button#contrast-button, #contrast-button button, #contrast-button"
                );
                if (button) {
                    button.setAttribute("aria-pressed", String(enabled));
                    button.textContent = enabled ? "Standard contrast" : "High contrast";
                }
            }
            """,
            queue=False,
        )

        read_aloud.click(
            None,
            chatbot,
            None,
            js="""
            (history) => {
                const messages = Array.isArray(history) ? history : [];
                const textFrom = (value) => {
                    if (value === null || value === undefined) return "";
                    if (typeof value === "string") return value;
                    if (Array.isArray(value)) return value.map(textFrom).filter(Boolean).join(" ");
                    if (typeof value === "object") {
                        if (typeof value.content === "string") return value.content;
                        if (typeof value.text === "string") return value.text;
                        if (typeof value.value === "string") return value.value;
                        return "";
                    }
                    return String(value);
                };
                let latestText = "";
                for (const message of [...messages].reverse()) {
                    if (Array.isArray(message) && message.length > 1) {
                        latestText = textFrom(message[1]);
                    } else if (message && message.role === "assistant") {
                        latestText = textFrom(message.content);
                    }
                    if (latestText) break;
                }
                if (!latestText || !("speechSynthesis" in window)) return;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(new SpeechSynthesisUtterance(latestText));
            }
            """,
            queue=False,
        )

        msg.submit(
            user,
            inputs=user_inputs,
            outputs=user_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            bot,
            inputs=bot_inputs,
            outputs=[chatbot, response_status],
            show_progress="hidden",
        ).then(
            save_participant_history,
            inputs=[chatbot, participant_state],
            outputs=None,
            queue=False,
        )

        send.click(
            user,
            inputs=user_inputs,
            outputs=user_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            bot,
            inputs=bot_inputs,
            outputs=[chatbot, response_status],
            show_progress="hidden",
        ).then(
            save_participant_history,
            inputs=[chatbot, participant_state],
            outputs=None,
            queue=False,
        )

        clear.click(
            reset_backend_chat,
            inputs=[api_base, greeting_box, session_id],
            outputs=[chatbot, session_id],
            queue=False,
        ).then(
            lambda: ("", gr.update(value="", visible=False)),
            None,
            [msg, response_status],
            queue=False,
        )

        download.click(
            export_transcript,
            inputs=chatbot,
            outputs=download,
        )

        for button, topic in zip(starter_buttons, PRE_TEST_TOPICS):
            button.click(
                lambda selected_topic=topic: selected_topic,
                None,
                msg,
                queue=False,
            )

    return demo


# Preserve `gradio ui_gradio.py` reload support without building a second app
# during a normal CLI launch or when importing helper functions in tests.
_reload_demo = None


def __getattr__(name):
    if name != "demo":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    global _reload_demo
    if _reload_demo is None:
        _reload_demo = build_ui(
            os.environ.get("MND_API_BASE", "http://127.0.0.1:8000"),
            os.environ.get("MND_MODEL", "mnd-local-agent"),
        )
    return _reload_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", "--api_base", dest="api_base", default=os.environ.get("MND_API_BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("MND_MODEL", "mnd-local-agent"))
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "8080")))
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes", "on"},
        help="Create a temporary public Gradio share link.",
    )
    args, _ = parser.parse_known_args()

    demo = build_ui(args.api_base, args.model)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="slate",
            radius_size="md",
            text_size="md",
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
