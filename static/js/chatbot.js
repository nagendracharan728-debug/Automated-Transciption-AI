async function sendQuestion() {
    const questionInput = document.getElementById("questionInput");
    const question = questionInput.value.trim();
    if(!question) return;

    const chatBox = document.getElementById("chatBox");
    const userMsg = document.createElement("div");
    userMsg.className = "chat-user";
    userMsg.innerText = "You: " + question;
    chatBox.appendChild(userMsg);

    questionInput.value = "";

    try {
        const response = await fetch("/ask_project_ai", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question})
        });

        // Make sure the response is JSON
        const data = await response.json();

        const aiMsg = document.createElement("div");
        aiMsg.className = "chat-ai";
        aiMsg.innerText = "AI: " + data.answer;
        chatBox.appendChild(aiMsg);

        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (err) {
        console.error(err);
        alert("Error communicating with the chatbot.");
    }
}