document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    function formatMessage(text) {
        // Safety check
        if (!text) return '';

        let formattedText = text;

        // First, escape any HTML to prevent XSS
        formattedText = formattedText
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');

        // Store numbered list items temporarily
        const numberedListItems = [];
        formattedText = formattedText.replace(/^\d+\.\s+(.+)$/gm, (match, content) => {
            numberedListItems.push(content);
            return `{{NUMBERED_LIST_ITEM_${numberedListItems.length - 1}}}`;
        });

        // Store bulleted list items temporarily
        const bulletedListItems = [];
        formattedText = formattedText.replace(/^[-•]\s+(.+)$/gm, (match, content) => {
            bulletedListItems.push(content);
            return `{{BULLETED_LIST_ITEM_${bulletedListItems.length - 1}}}`;
        });

        // Handle Markdown links
        formattedText = formattedText.replace(
            /\[([^\]]+)\]\(([^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        // Handle plain URLs (but avoid already processed links)
        formattedText = formattedText.replace(
            /(?<!href=")(https?:\/\/[^\s<]+)/g,
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        // Restore numbered list items with proper HTML
        formattedText = formattedText.replace(/{{NUMBERED_LIST_ITEM_(\d+)}}/g, (match, index) => {
            return `<div class="list-item numbered"><span class="number">${parseInt(index) + 1}.</span> ${numberedListItems[index]}</div>`;
        });

        // Restore bulleted list items with proper HTML
        formattedText = formattedText.replace(/{{BULLETED_LIST_ITEM_(\d+)}}/g, (match, index) => {
            return `<div class="list-item bulleted">• ${bulletedListItems[index]}</div>`;
        });

        // Handle paragraph breaks (double newlines)
        formattedText = formattedText.replace(/\n\n+/g, '</p><p>');

        // Handle single newlines
        formattedText = formattedText.replace(/\n/g, '<br>');

        // Wrap in paragraphs if not already wrapped
        if (!formattedText.startsWith('<p>')) {
            formattedText = `<p>${formattedText}</p>`;
        }

        return formattedText;
    }

    function addMessage(message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        if (isUser) {
            messageContent.textContent = message;
        } else {
            messageContent.innerHTML = formatMessage(message);
        }

        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function disableInput() {
        userInput.disabled = true;
        sendButton.disabled = true;
        document.getElementById('endChatButton').disabled = true;
    }

    function enableInput() {
        userInput.disabled = false;
        sendButton.disabled = false;
        document.getElementById('endChatButton').disabled = false;
        userInput.focus();
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
    
        addMessage(message, true);
        userInput.value = '';
        disableInput();
    
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot';
        const loadingContent = document.createElement('div');
        loadingContent.className = 'message-content loading';
        loadingContent.textContent = 'Thinking...';
        loadingDiv.appendChild(loadingContent);
        chatMessages.appendChild(loadingDiv);
    
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });
    
            const data = await response.json();
            chatMessages.removeChild(loadingDiv);
    
            if (data.error) {
                addMessage("I apologize, but I encountered an error. Please try again.", false);
            } else {
                addMessage(data.response, false);
    
                // Check if this is an exit intent
                if (data.is_exit === true) {
                    handleExit();
                }
            }
        } catch (error) {
            console.error('Error:', error);
            chatMessages.removeChild(loadingDiv);
            addMessage("I apologize, but I encountered an error. Please try again.", false);
        }
    
        enableInput();
    }    

    function handleExit() {
        showRatingPrompt();
        disableInput();
    }

    function showRatingPrompt() {
        const ratingContainer = document.getElementById('rating-container');
        ratingContainer.style.display = 'block';
        
        // Scroll to the rating container
        ratingContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // Function to handle rating submission
    async function submitRating(rating) {
        try {
            const response = await fetch('/submit_rating', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rating: rating })
            });
            
            const data = await response.json();
            alert(data.message);
            document.getElementById('rating-container').style.display = 'none';
            
            // Add a final thank you message
            addMessage("Thank you for your feedback! Have a great day!", false);
            
            // Keep the chat disabled after rating
            disableInput();
            
        } catch (error) {
            console.error('Error submitting rating:', error);
            alert('Error submitting rating. Please try again.');
        }
    }

    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    document.getElementById('endChatButton').addEventListener('click', function() {
        handleExit();
    });

    // Add event listeners for rating buttons
    const ratingButtons = document.querySelectorAll('#rating-container button');
    ratingButtons.forEach(button => {
        button.addEventListener('click', function() {
            submitRating(parseInt(this.textContent));
        });
    });
});