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
dropArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFileName.textContent = file.name;
        // Automatically trigger upload when file is selected
        uploadButton.click();
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

            if (data.success) {
                showAlert('File uploaded successfully', 'success');
                // Switch to analysis section
                document.querySelector('[data-section="analysis"]').click();
                // Store document ID for future use
                window.currentDocId = data.document_id;
                // Load document analysis
                loadDocumentAnalysis(data.document_id);
            } else {
                showAlert(data.error || 'Upload failed', 'error');
            }
        } else {
            showAlert(data.error, 'error');
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

document.addEventListener('DOMContentLoaded', function() {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetSection = link.getAttribute('data-section');
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Show target section, hide others
            sections.forEach(section => {
                section.style.display = section.id === `${targetSection}-section` ? 'block' : 'none';
            });
        });
    });

    // File Upload
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const uploadLoading = document.getElementById('upload-loading');
    const uploadAlert = document.getElementById('upload-alert');

    // Drag and drop handlers
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

    function highlight(e) {
        dropArea.classList.add('highlight');
    }

    function unhighlight(e) {
        dropArea.classList.remove('highlight');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                document.getElementById('selected-file-name').textContent = file.name;
                uploadButton.disabled = false;
            } else {
                showAlert('Please select a PDF file', 'error');
            }
        }
    }

    uploadButton.addEventListener('click', async function() {
        const file = fileInput.files[0];
        if (!file) {
            showAlert('Please select a file first', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            uploadLoading.style.display = 'block';
            uploadButton.disabled = true;

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                showAlert('File uploaded successfully', 'success');
                // Switch to analysis section
                document.querySelector('[data-section="analysis"]').click();
                // Store document ID for future use
                window.currentDocId = data.document_id;
                // Load document analysis
                loadDocumentAnalysis(data.document_id);
            } else {
                showAlert(data.error || 'Upload failed', 'error');
            }
        } catch (error) {
            showAlert('Error uploading file', 'error');
        } finally {
            uploadLoading.style.display = 'none';
            uploadButton.disabled = false;
        }
    });

    // Question Answering
    const questionInput = document.getElementById('question-input');
    const askButton = document.getElementById('ask-button');
    const questionLoading = document.getElementById('question-loading');
    const questionAlert = document.getElementById('question-alert');
    const answerSection = document.getElementById('answer-section');
    const answerText = document.getElementById('answer-text');

    askButton.addEventListener('click', async function() {
        const question = questionInput.value.trim();
        if (!question) {
            showAlert('Please enter a question', 'error', questionAlert);
            return;
        }

        try {
            questionLoading.style.display = 'block';
            askButton.disabled = true;

            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    document_id: window.currentDocId
                })
            });

            const data = await response.json();

            if (data.success) {
                answerText.innerHTML = data.answer;
                answerSection.style.display = 'block';
            } else {
                showAlert(data.error || 'Failed to get answer', 'error', questionAlert);
            }
        } catch (error) {
            showAlert('Error processing question', 'error', questionAlert);
        } finally {
            questionLoading.style.display = 'none';
            askButton.disabled = false;
        }
    });

    // Summary Generation
    const summaryType = document.getElementById('summary-type');
    const generateSummary = document.getElementById('generate-summary');
    const summaryLoading = document.getElementById('summary-loading');
    const summaryContent = document.getElementById('summary-content');
    const summaryText = document.getElementById('summary-text');

    generateSummary.addEventListener('click', async function() {
        if (!window.currentDocId) {
            showAlert('Please upload a document first', 'error');
            return;
        }

        try {
            summaryLoading.style.display = 'block';
            generateSummary.disabled = true;

            const response = await fetch(`/summarize/${window.currentDocId}?type=hybrid`, {
                method: 'GET'
            });

            const data = await response.json();

            if (data.success) {
                // Only show summary if it's not a fallback
                if (data.method !== 'fallback') {
                    summaryText.innerHTML = data.summary;
                    
                    // Display key points if available
                    const keyPointsList = document.getElementById('key-points');
                    keyPointsList.innerHTML = ''; // Clear existing points
                    if (data.key_points && data.key_points.length > 0) {
                        data.key_points.forEach(point => {
                            const li = document.createElement('li');
                            li.className = 'list-group-item';
                            li.textContent = point;
                            keyPointsList.appendChild(li);
                        });
                    }
                    
                    summaryContent.style.display = 'block';
                } else {
                    // Hide the summary section if it's a fallback
                    summaryContent.style.display = 'none';
                }
            } else {
                showAlert(data.error || 'Failed to generate summary', 'error');
            }
        } catch (error) {
            showAlert('Error generating summary', 'error');
        } finally {
            summaryLoading.style.display = 'none';
            generateSummary.disabled = false;
        }
    });

    // Research Connections
    const timeRange = document.getElementById('time-range');
    const connectionType = document.getElementById('connection-type');
    const updateConnections = document.getElementById('update-connections');
    const connectionsVisualization = document.getElementById('connections-visualization');

    updateConnections.addEventListener('click', async function() {
        if (!window.currentDocId) {
            showAlert('Please upload a document first', 'error');
            return;
        }

        try {
            const response = await fetch('/citation-network', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    document_id: window.currentDocId,
                    time_range: timeRange.value,
                    connection_type: connectionType.value
                })
            });

            const data = await response.json();

            if (data.success) {
                // Initialize visualization using vis.js
                const container = connectionsVisualization;
                const nodes = new vis.DataSet(data.nodes);
                const edges = new vis.DataSet(data.edges);
                
                const network = new vis.Network(container, { nodes, edges }, {
                    nodes: {
                        shape: 'dot',
                        size: 16
                    },
                    edges: {
                        arrows: {
                            to: { enabled: true, scaleFactor: 1 }
                        }
                    },
                    physics: {
                        stabilization: false,
                        barnesHut: {
                            gravitationalConstant: -80000,
                            springConstant: 0.001,
                            springLength: 200
                        }
                    }
                });
            } else {
                showAlert(data.error || 'Failed to load connections', 'error');
            }
        } catch (error) {
            showAlert('Error loading connections', 'error');
        }
    });

    // Utility Functions
    function showAlert(message, type, element = uploadAlert) {
        element.textContent = message;
        element.className = `alert alert-${type}`;
        element.style.display = 'block';
        
        setTimeout(() => {
            element.style.display = 'none';
        }, 5000);
    }
});

// Add document analysis functionality
function loadDocumentAnalysis(docId) {
    fetch(`/document-analysis/${docId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Display figures
                const figuresContainer = document.getElementById('figures-content');
                if (data.figures && data.figures.length > 0) {
                    figuresContainer.innerHTML = data.figures.map(figure => `
                        <div class="figure-card">
                            <img src="${figure.url}" alt="${figure.caption || 'Figure'}">
                            <div class="figure-caption">
                                <strong>Page ${figure.page}</strong>
                                <p>${figure.caption || 'No caption available'}</p>
                            </div>
                        </div>
                    `).join('');
                } else {
                    figuresContainer.innerHTML = '<p class="text-muted">No figures found in this document.</p>';
                }

                // Display tables
                const tablesContainer = document.getElementById('tables-content');
                if (data.tables && data.tables.length > 0) {
                    tablesContainer.innerHTML = data.tables.map(table => `
                        <div class="table-container">
                            <div class="table-caption">
                                <strong>Page ${table.page}</strong>
                                <p>${table.caption || 'No caption available'}</p>
                            </div>
                            <div class="table-responsive">
                                ${table.html}
                            </div>
                        </div>
                    `).join('');
                } else {
                    tablesContainer.innerHTML = '<p class="text-muted">No tables found in this document.</p>';
                }

                // Display citations
                const citationsContainer = document.getElementById('citations-content');
                if (data.citations && data.citations.length > 0) {
                    citationsContainer.innerHTML = data.citations.map(citation => `
                        <div class="citation-item">
                            <h5>${citation.title || 'Untitled'}</h5>
                            <p class="citation-authors">${citation.authors.join(', ')}</p>
                            <p class="citation-year">${citation.year}</p>
                        </div>
                    `).join('');
                } else {
                    citationsContainer.innerHTML = '<p class="text-muted">No citations found in this document.</p>';
                }
            } else {
                showAlert('Error loading document analysis', 'danger');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('Error loading document analysis', 'danger');
        });
}