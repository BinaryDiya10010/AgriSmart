/**
 * AgriSmart AI Platform - Main JavaScript
 * All client-side interactivity and animations
 */

// ==================== CAROUSEL FUNCTIONALITY ====================

/**
 * Initialize image carousel with auto-advance and manual navigation
 */
function initCarousel(containerSelector) {
    const container = document.querySelector(containerSelector);
    if (!container) return;

    const slides = container.querySelectorAll('.carousel-slide');
    if (slides.length === 0) return;

    let currentSlide = 0;
    let autoAdvanceInterval;

    // Show specific slide
    function showSlide(index) {
        slides.forEach((slide, i) => {
            slide.classList.toggle('active', i === index);
        });
        currentSlide = index;
    }

    // Next slide
    function nextSlide() {
        const next = (currentSlide + 1) % slides.length;
        showSlide(next);
    }

    // Auto-advance every 5 seconds
    function startAutoAdvance() {
        if (autoAdvanceInterval) clearInterval(autoAdvanceInterval);
        autoAdvanceInterval = setInterval(nextSlide, 5000);
    }

    // Initialize
    showSlide(0);
    startAutoAdvance();
}


// ==================== FORM VALIDATION ====================

/**
 * Advanced client-side form validation
 */
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return true;

    let isValid = true;
    const requiredFields = form.querySelectorAll('[required]');

    requiredFields.forEach(field => {
        const value = field.value.trim();

        if (!value) {
            field.style.borderColor = 'var(--error-red)';
            isValid = false;
        } else {
            field.style.borderColor = '';
        }
    });

    return isValid;
}


// ==================== SLIDER FUNCTIONALITY ====================

/**
 * Initialize range slider with real-time display
 */
function initSlider(sliderId, displayId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);

    if (!slider || !display) return;

    slider.addEventListener('input', function () {
        display.textContent = this.value;

        // Update color based on pH value (if pH slider)
        if (sliderId === 'ph') {
            const ph = parseFloat(this.value);
            if (ph < 6.0) {
                display.style.color = 'var(--warning-yellow)';
            } else if (ph > 7.5) {
                display.style.color = 'var(--sky-blue)';
            } else {
                display.style.color = 'var(--success-green)';
            }
        }
    });
}


// ==================== FILE UPLOAD PREVIEW ====================

/**
 * Image upload preview with validation
 */
function initFileUpload(inputId, previewId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);

    if (!input || !preview) return;

    input.addEventListener('change', function (e) {
        const file = e.target.files[0];

        if (!file) {
            preview.innerHTML = '';
            return;
        }

        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            preview.innerHTML = '<p style="color: var(--error-red);">❌ Invalid file type</p>';
            input.value = '';
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            preview.innerHTML = '<p style="color: var(--error-red);">❌ File too large (max 16MB)</p>';
            input.value = '';
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.innerHTML = `
                <div style="position: relative; display: inline-block;">
                    <img src="${e.target.result}" 
                         style="max-width: 100%; max-height: 400px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <div style="position: absolute; top: 10px; right: 10px; background: var(--sage-green); color: white; padding: 6px 12px; border-radius: 6px;">
                        ✓ Ready
                    </div>
                </div>
            `;
        };
        reader.readAsDataURL(file);
    });
}


// ==================== ACCORDION FUNCTIONALITY ====================

/**
 * Initialize accordion toggles
 */
function initAccordions() {
    const accordions = document.querySelectorAll('.accordion-item details');
    // Details/summary handle themselves automatically
}


// ==================== SMOOTH SCROLLING ====================

/**
 * Smooth scroll for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return;

            e.preventDefault();
            const target = document.querySelector(href);

            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}


// ==================== COMPATIBILITY BAR ANIMATION ====================

/**
 * Animate compatibility/progress bars
 */
function animateCompatibilityBars() {
    const bars = document.querySelectorAll('.compatibility-fill');

    bars.forEach(bar => {
        const targetWidth = bar.getAttribute('data-width') || bar.style.width;
        bar.style.width = '0%';

        setTimeout(() => {
            bar.style.width = targetWidth;
        }, 100);
    });
}


// ==================== FLASH MESSAGE AUTO-DISMISS ====================

/**
 * Auto-dismiss flash messages after 5 seconds
 */
function initFlashMessages() {
    const alerts = document.querySelectorAll('.alert');

    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    });
}


// ==================== CHART.JS INTEGRATION ====================

/**
 * Initialize price trend chart using Chart.js
 */
function initPriceTrendChart(canvasId, cropName, priceData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') {
        console.log('Chart.js not loaded or canvas not found');
        return;
    }

    const ctx = canvas.getContext('2d');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: priceData.labels,
            datasets: [{
                label: cropName + ' Price (₹/Quintal)',
                data: priceData.values,
                borderColor: 'rgba(184, 212, 184, 1)',
                backgroundColor: 'rgba(184, 212, 184, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}


// ==================== INITIALIZATION ====================

/**
 * Initialize all JavaScript features on page load
 */
document.addEventListener('DOMContentLoaded', function () {
    // Carousel
    initCarousel('#heroCarousel');

    // Sliders
    initSlider('ph', 'phValue');

    // File uploads
    initFileUpload('imageUpload', 'imagePreview');

    // Accordions
    initAccordions();

    // Smooth scrolling
    initSmoothScroll();

    // Animate bars
    setTimeout(animateCompatibilityBars, 200);

    // Flash messages
    initFlashMessages();

    console.log('✅ AgriSmart AI - All JavaScript features initialized');
});
