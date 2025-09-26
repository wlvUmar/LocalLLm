class SimpleChat {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.messages = document.getElementById('messages');
        this.status = document.getElementById('status');
        this.modelSelect = document.getElementById('modelSelect');
        this.switchBtn = document.getElementById('switchBtn');
        this.currentModel = document.getElementById('currentModel');
        this.alert = document.getElementById('alert');
        this.loadMore = document.getElementById('loadMore');
        
        this.isProcessing = false;
        this.messageList = [];
        this.offset = 0;
        this.limit = 50;
        
        this.init();
    }
    
    async init() {
        this.setupEvents();
        await this.updateStatus();
        await this.loadMessages();
    }
    
    setupEvents() {
        this.sendBtn.onclick = () => this.sendMessage();
        this.switchBtn.onclick = () => this.switchModel();
        this.loadMore.onclick = () => this.loadMoreMessages();
        
        this.messageInput.onkeydown = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        };
        
        this.messageInput.oninput = () => this.updateSendBtn();
        this.updateSendBtn();
    }
    
    updateSendBtn() {
        this.sendBtn.disabled = !this.messageInput.value.trim() || this.isProcessing;
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.server_ready) {
                this.status.textContent = 'Ready';
                this.status.className = 'status ready';
                this.currentModel.textContent = data.current_model || 'No model';
                
                if (data.available_models) {
                    this.modelSelect.innerHTML = '<option value="">Select Model...</option>';
                    data.available_models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        option.selected = model === data.current_model;
                        this.modelSelect.appendChild(option);
                    });
                }
            } else {
                this.status.textContent = 'Not Ready';
                this.status.className = 'status error';
            }
        } catch (error) {
            this.status.textContent = 'Error';
            this.status.className = 'status error';
            console.error('Status update failed:', error);
        }
    }
    
    async switchModel() {
        if (!this.modelSelect.value) {
            this.showAlert('Please select a model', 'error');
            return;
        }
        
        this.switchBtn.disabled = true;
        this.switchBtn.textContent = 'Switching...';
        
        try {
            const response = await fetch('/api/switch-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: this.modelSelect.value })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showAlert(result.message, 'success');
                this.currentModel.textContent = this.modelSelect.value;
            } else {
                this.showAlert(result.detail || 'Switch failed', 'error');
            }
        } catch (error) {
            this.showAlert('Network error', 'error');
            console.error('Model switch failed:', error);
        } finally {
            this.switchBtn.disabled = false;
            this.switchBtn.textContent = 'Switch';
        }
    }
    
    async loadMessages() {
        try {
            const response = await fetch(`/api/messages?limit=${this.limit}&offset=0`);
            const data = await response.json();
            
            this.messageList = data.messages || [];
            this.offset = this.messageList.length;
            
            if (this.messageList.length === 0) {
                this.addMessage('assistant', 'Hello! How can I help you today?', Date.now());
            } else {
                this.renderMessages();
            }
            
            this.scrollToBottom();
        } catch (error) {
            console.error('Failed to load messages:', error);
            this.addMessage('assistant', 'Hello! How can I help you today?', Date.now());
        }
    }
    
    async loadMoreMessages() {
        if (this.loadMore.classList.contains('loading')) return;
        
        this.loadMore.classList.add('loading');
        this.loadMore.textContent = 'Loading...';
        
        try {
            const response = await fetch(`/api/messages?limit=${this.limit}&offset=${this.offset}`);
            const data = await response.json();
            
            const newMessages = data.messages || [];
            if (newMessages.length > 0) {
                const scrollHeight = this.messages.scrollHeight;
                this.messageList = [...newMessages, ...this.messageList];
                this.offset += newMessages.length;
                
                this.renderMessages();
                
                const newScrollHeight = this.messages.scrollHeight;
                this.messages.scrollTop += (newScrollHeight - scrollHeight);
            } else {
                this.loadMore.style.display = 'none';
            }
        } catch (error) {
            this.showAlert('Failed to load more messages', 'error');
            console.error('Load more failed:', error);
        } finally {
            this.loadMore.classList.remove('loading');
            this.loadMore.textContent = 'Load more messages';
        }
    }
    
    renderMessages() {
        this.messages.innerHTML = '';
        this.messageList.forEach(msg => {
            this.addMessageToDOM(msg.role, msg.content, msg.timestamp);
        });
        
        if (this.offset > 0) {
            this.loadMore.style.display = 'block';
        }
    }
    
    addMessage(role, content, timestamp) {
        this.messageList.push({ role, content, timestamp });
        this.addMessageToDOM(role, content, timestamp);
        this.scrollToBottom();
    }
    
    addMessageToDOM(role, content, timestamp) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        
        const time = new Date(timestamp).toLocaleTimeString();
        const sender = role === 'user' ? 'You' : 'Assistant';
        
        div.innerHTML = `
            <div class="message-header">${sender} - ${time}</div>
            <div class="message-content">${content}</div>
        `;
        
        this.messages.appendChild(div);
    }
    
    showTyping() {
        const div = document.createElement('div');
        div.className = 'message typing';
        div.id = 'typing';
        div.innerHTML = `
            <div class="message-header">Assistant - typing...</div>
            <div class="message-content">...</div>
        `;
        this.messages.appendChild(div);
        this.scrollToBottom();
    }
    
    hideTyping() {
        const typing = document.getElementById('typing');
        if (typing) typing.remove();
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;
        
        this.isProcessing = true;
        this.updateSendBtn();
        
        this.addMessage('user', message, Date.now());
        this.messageInput.value = '';
        this.showTyping();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, stream: true })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            this.hideTyping();
            
            // Create assistant message container immediately
            const assistantMsg = { role: 'assistant', content: '', timestamp: Date.now() };
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message assistant';
            msgDiv.innerHTML = `
                <div class="message-header">Assistant - ${new Date().toLocaleTimeString()}</div>
                <div class="message-content"></div>
            `;
            this.messages.appendChild(msgDiv);
            const contentDiv = msgDiv.querySelector('.message-content');
            
            // Stream the response token by token
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    // Decode and add to buffer
                    buffer += decoder.decode(value, { stream: true });
                    console.log('Buffer:', JSON.stringify(buffer)); // Debug log
                    
                    // Process complete SSE messages
                    let newlineIndex;
                    while ((newlineIndex = buffer.indexOf('\n\n')) !== -1) {
                        const message = buffer.slice(0, newlineIndex);
                        buffer = buffer.slice(newlineIndex + 2);
                        
                        console.log('Processing SSE message:', JSON.stringify(message)); // Debug log
                        
                        if (message.startsWith('data: ')) {
                            const token = message.slice(6);
                            console.log('Extracted token:', JSON.stringify(token)); // Debug log
                            
                            if (token) {
                                // Add token immediately to DOM
                                assistantMsg.content += token;
                                contentDiv.textContent = assistantMsg.content;
                                this.scrollToBottom();
                            }
                        }
                    }
                }
            } catch (streamError) {
                console.error('Streaming error:', streamError);
                contentDiv.textContent = assistantMsg.content + '\n[Stream interrupted]';
            }
            
            // Add final message to list
            this.messageList.push(assistantMsg);
            
        } catch (error) {
            this.hideTyping();
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.', Date.now());
            this.showAlert('Failed to send message: ' + error.message, 'error');
            console.error('Send message failed:', error);
        } finally {
            this.isProcessing = false;
            this.updateSendBtn();
            this.messageInput.focus();
        }
    }
    
    showAlert(message, type = '') {
        this.alert.textContent = message;
        this.alert.className = `alert ${type}`;
        this.alert.style.display = 'block';
        
        setTimeout(() => {
            this.alert.style.display = 'none';
        }, 5000);
    }
    
    scrollToBottom() {
        this.messages.scrollTop = this.messages.scrollHeight;
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new SimpleChat());
} else {
    new SimpleChat();
}