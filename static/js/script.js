// Initialize webcam
let video = document.getElementById('video');
let canvas = document.createElement('canvas');
let ctx = canvas.getContext('2d');
let stream = null;
let currentImage = null;

// Elements
const captureBtn = document.getElementById('capture');
const analyzeBtn = document.getElementById('analyzeBtn');
const imageUpload = document.getElementById('imageUpload');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const preview = document.getElementById('preview');
const resultIcon = document.getElementById('resultIcon');
const resultText = document.getElementById('resultText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const resultMessage = document.getElementById('resultMessage');
const uploadArea = document.querySelector('.upload-area');

// Init
async function init() {
    // FIXED: Add video ready check
    video.addEventListener('loadedmetadata', function() {
        console.log('Video ready - dimensions:', video.videoWidth, 'x', video.videoHeight);
    }, { once: true });
    
    video.addEventListener('loadeddata', function() {
        console.log('Video loaded data ready for capture');
    });
    
    try {
        console.log('Requesting camera access...');
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, facingMode: 'user' } 
        });
        video.srcObject = stream;
        console.log('Camera stream started successfully');
    } catch(err) {
        console.error('Webcam access failed:', err);
        // Show user-friendly message instead of hiding button
        captureBtn.innerHTML = '<i class="fas fa-camera"></i> Camera Unavailable';
        captureBtn.disabled = true;
        captureBtn.title = 'Camera access denied or unavailable. Use file upload instead.';
    }
    
    // Drag & drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    imageUpload.addEventListener('change', handleFileSelect);
    
// captureBtn.addEventListener('click', captureImage); // Replaced by enhanced handler
    analyzeBtn.addEventListener('click', analyzeImage);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageUpload.files = files;
        handleFileSelect();
    }
}

function handleFileSelect() {
    const file = imageUpload.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImage = e.target.result;
            preview.src = currentImage;
            preview.style.display = 'block';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

async function captureImage() {
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.error('Video not ready for capture');
        showNotification('❌ Video not ready. Please wait...');
        return;
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    currentImage = canvas.toDataURL('image/jpeg', 0.8);
    preview.src = currentImage;
    preview.style.display = 'block';
    
    // Capture feedback
    showNotification('✅ Image captured successfully!');
    // Don't disable - use visual state only for recapture toggle
    captureBtn.innerHTML = '<i class="fas fa-redo"></i> Recapture';
    captureBtn.classList.add('btn-secondary');
    captureBtn.classList.remove('btn-primary');
    preview.classList.add('captured');
    
    analyzeBtn.disabled = false;
}

function showNotification(message) {
    // Create toast
    let toast = document.querySelector('.toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'toast';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 2000);
}

// Enhanced capture handler - checks button text for state
captureBtn.addEventListener('click', function(e) {
    if (this.innerHTML.includes('Recapture')) {
        // Recapture logic - reset
        this.innerHTML = '<i class="fas fa-camera"></i> Capture';
        this.classList.add('btn-primary');
        this.classList.remove('btn-secondary');
        preview.classList.remove('captured');
        currentImage = null;
        analyzeBtn.disabled = true;
        preview.src = '';
        preview.style.display = 'none';
        preview.classList.add('hidden');
        showNotification('📹 Ready to capture new image');
    } else {
        captureImage();
    }
});

async function analyzeImage() {
    if (!currentImage) {
        showNotification('❌ No image to analyze');
        return;
    }
    console.log('Starting analysis...');
    
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    analyzeBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('base64', currentImage);
        
        console.log('Sending to /predict...');
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            alert('Error: ' + result.error);
            return;
        }
        
        displayResults(result);
    } catch(err) {
        alert('Analysis failed: ' + err.message);
    } finally {
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

function displayResults(result) {
    // FIXED: Show analyzed image in results preview above results
    const resultPreview = document.getElementById('resultPreview');
    if (resultPreview && currentImage) {
        resultPreview.src = currentImage;
        resultPreview.style.display = 'block';
    }
    
    const isJaundice = result.result === 'jaundice';
    const conf = result.confidence;
    
    // Icon & Text
    resultIcon.innerHTML = isJaundice ? 
        '<i class="fas fa-exclamation-triangle"></i>' : 
        '<i class="fas fa-check-circle"></i>';
    
    resultText.textContent = isJaundice ? 'Jaundice Detected' : 'Normal';
    
    // Confidence
    confidenceText.textContent = `${Math.round(conf * 100)}% Confidence`;
    
    const fill = confidenceBar.querySelector('.confidence-fill') || 
                 document.createElement('div');
    fill.className = `confidence-fill ${isJaundice ? 'confidence-jaundice' : 'confidence-normal'}`;
    fill.style.width = `${conf * 100}%`;
    confidenceBar.innerHTML = '';
    confidenceBar.appendChild(fill);
    
    // Message
    resultMessage.innerHTML = result.message + 
        '<br><small>This is AI analysis only. Consult medical professional.</small>';
    
    // Theme
    results.className = `results ${isJaundice ? 'result-jaundice' : 'result-normal'}`;
    results.classList.remove('hidden');
}

// Start
init();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
