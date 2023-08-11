document.addEventListener("DOMContentLoaded", () => {
    const chatLog = document.getElementById("chat-log");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    sendButton.addEventListener("click", async () => {
        const userMessage = userInput.value;
        if (!userMessage) return;

        // Append user message to the chat log
        chatLog.innerHTML += `<div class="message user">${userMessage}</div>`;

        // Clear user input
        userInput.value = "";

        // Send user input to the server and get the bot's response
        const response = await fetch("/get_response", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body: `user_input=${encodeURIComponent(userMessage)}`,
        });
        console.log(response)
        const data = await response.json();

        // Append bot response to the chat log
        chatLog.innerHTML += `<div class="message bot">${data.bot_response}</div>`;

        // Scroll to the bottom of the chat log
        chatLog.scrollTop = chatLog.scrollHeight;
    });
});
