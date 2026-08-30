let conversationId = null;

function addMessage(role, content) {
    const messages = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.textContent = content;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
}

async function sendMessage() {
    const input = document.getElementById("userInput");
    const btn = document.getElementById("sendBtn");
    const message = input.value.trim();
    if (!message) return;

    addMessage("user", message);
    input.value = "";
    input.style.height = "auto";

    btn.disabled = true;
    const loadingDiv = addMessage("assistant loading", "Thinking...");

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, conversation_id: conversationId }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            const msg = data.detail || data.response || `Server error ${res.status}`;
            loadingDiv.textContent = msg;
            loadingDiv.classList.remove("loading");
            loadingDiv.classList.add("error");
            return;
        }
        conversationId = data.conversation_id || conversationId;
        const reply = data.response || data.detail || "No reply from server.";
        loadingDiv.textContent = reply;
        loadingDiv.classList.remove("loading");
    } catch (err) {
        loadingDiv.textContent = "Error: could not reach server. Check /health - model may be loading (355MB). Try again in 30s. " + (err.message || "");
        loadingDiv.classList.remove("loading");
        loadingDiv.classList.add("error");
    }

    btn.disabled = false;
    input.focus();
}

function newChat() {
    conversationId = null;
    document.getElementById("messages").innerHTML =
        '<div class="message assistant">Hello! I\'m your AI assistant. Ask me anything or let me help you solve a problem.</div>';
}

function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// Auto-resize textarea
document.getElementById("userInput").addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
});
