// DOM elements
const fileInput = document.getElementById('file-input');
const dropArea = document.getElementById('drop-area');
const selectedFileName = document.getElementById('selected-file-name');
const uploadButton = document.getElementById('upload-button');
const uploadAlert = document.getElementById('upload-alert');
const uploadLoading = document.getElementById('upload-loading');
const questionCard = document.getElementById('question-card');
const questionInput = document.getElementById('question-input');
const askButton = document.getElementById('ask-button');
const questionAlert = document.getElementById('question-alert');
const questionLoading = document.getElementById('question-loading');
const answerSection = document.getElementById('answer-section');
const answerText = document.getElementById('answer-text');

// Zotero elements (may be null if not enabled)
const zoteroCard = document.getElementById('zotero-card');
const zoteroCollections = document.getElementById('zotero-collections');
const addToZoteroButton = document.getElementById('add-to-zotero-button');
const zoteroAlert = document.getElementById('zotero-alert');

// Handle file selection
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFileName.textContent = file.name;
    } else {
        selectedFileName.textContent = '';
    }
});

// Handle drag and drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropArea.classList.add('highlighted');
}

function unhighlight() {
    dropArea.classList.remove('highlighted');
}

dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    fileInput.files = dt.files;
    if (file) {
        selectedFileName.textContent = file.name;
    }
}

// Handle file upload
uploadButton.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        showAlert(uploadAlert, 'Please select a PDF file first', 'error');
        return;
    }

    // Create form data
    const formData = new FormData();
    formData.append('file', file);

    // Show loading
    uploadLoading.style.display = 'block';
    uploadAlert.classList.add('hidden');

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showAlert(uploadAlert, data.message, 'success');
            questionCard.style.display = 'block';
            
            // Show Zotero card if it exists
            if (zoteroCard) {
                zoteroCard.style.display = 'block';
                // Load Zotero collections if needed
                loadZoteroCollections();
            }
            
            window.scrollTo({
                top: questionCard.offsetTop,
                behavior: 'smooth'
            });
        } else {
            showAlert(uploadAlert, data.error, 'error');
        }
    } catch (error) {
        showAlert(uploadAlert, 'An error occurred during upload', 'error');
        console.error(error);
    } finally {
        uploadLoading.style.display = 'none';
    }
});

// Handle asking a question
askButton.addEventListener('click', async () => {
    console.log("Ask button clicked");
    const question = questionInput.value.trim();
    if (!question) {
        showAlert(questionAlert, 'Please enter a question', 'error');
        return;
    }

    // Show loading
    console.log("Showing loading spinner");
    questionLoading.style.display = 'block';
    questionAlert.classList.add('hidden');
    answerSection.style.display = 'none';

    try {
        console.log("Sending request to /ask");
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        console.log("Response received:", response.status);
        const data = await response.json();
        console.log("Data:", data);

        if (response.ok) {
            answerText.textContent = data.answer;
            answerSection.style.display = 'block';
        } else {
            console.error("Error in ask request:", error);
            showAlert(questionAlert, data.error, 'error');
        }
    } catch (error) {
        showAlert(questionAlert, 'An error occurred while getting the answer', 'error');
        console.error(error);
    } finally {
        console.log("Hiding loading spinner");
        questionLoading.style.display = 'none';
    }
});

// Load Zotero collections
async function loadZoteroCollections() {
    if (!zoteroCollections) return;
    
    try {
        const response = await fetch('/zotero/collections');
        const data = await response.json();
        
        if (Array.isArray(data)) {
            // Clear existing options except the default
            while (zoteroCollections.options.length > 1) {
                zoteroCollections.remove(1);
            }
            
            // Add new options
            data.forEach(collection => {
                const option = document.createElement('option');
                option.value = collection.key;
                option.textContent = collection.data.name;
                zoteroCollections.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading Zotero collections:', error);
    }
}

// Handle adding to Zotero
if (addToZoteroButton) {
    addToZoteroButton.addEventListener('click', async () => {
        const title = document.getElementById('doc-title').value.trim();
        const authors = document.getElementById('doc-authors').value.trim().split(',').map(a => a.trim());
        const year = document.getElementById('doc-year').value.trim();
        const doi = document.getElementById('doc-doi').value.trim();
        const collectionKey = zoteroCollections.value;
        
        if (!title) {
            showAlert(zoteroAlert, 'Please enter a title', 'error');
            return;
        }
        
        try {
            const response = await fetch('/zotero/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    metadata: {
                        title,
                        authors,
                        year,
                        doi
                    },
                    collection_key: collectionKey || null
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showAlert(zoteroAlert, data.message, 'success');
            } else {
                showAlert(zoteroAlert, data.error, 'error');
            }
        } catch (error) {
            showAlert(zoteroAlert, 'Error adding to Zotero', 'error');
            console.error(error);
        }
    });
}

// Show alert message
function showAlert(element, message, type) {
    element.textContent = message;
    element.classList.remove('hidden', 'alert-error', 'alert-success');
    element.classList.add(`alert-${type}`);
}

// Allow pressing Enter to submit question
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        askButton.click();
    }
});