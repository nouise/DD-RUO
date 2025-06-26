// Main JavaScript for Joint Rate-Utility Optimization for Dataset Distillation
// Navbar scroll effect
window.addEventListener('scroll', function() {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});
// Theme toggle
const themeToggle = document.getElementById('themeToggle');
const body = document.body;
const currentTheme = localStorage.getItem('theme') || 'light';
body.setAttribute('data-theme', currentTheme);
themeToggle.textContent = currentTheme === 'dark' ? '☀️' : '🌙';
themeToggle.addEventListener('click', function() {
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    themeToggle.textContent = newTheme === 'dark' ? '☀️' : '🌙';
});
// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const offsetTop = target.offsetTop - 100;
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    });
});
// Add loading animation to external links
document.querySelectorAll('.paper-link').forEach(link => {
    link.addEventListener('click', function(e) {
        if (this.getAttribute('href') === '#') {
            e.preventDefault();
            const originalText = this.innerHTML;
            this.innerHTML = '<span class="loading"></span> Loading...';
            setTimeout(() => {
                this.innerHTML = originalText;
                alert('Link not configured. Please add the actual link before deployment.');
            }, 1000);
        }
    });
});
// Intersection Observer for section highlighting
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav-links a');
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const id = entry.target.getAttribute('id');
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${id}`) {
                    link.classList.add('active');
                }
            });
        }
    });
}, {
    threshold: 0.3,
    rootMargin: '-100px 0px -100px 0px'
});
sections.forEach(section => {
    observer.observe(section);
});
// 轮播图逻辑独立
window.addEventListener('DOMContentLoaded', function() {
  const carouselDir = 'data/img/distilled_img/';
  const imgFiles = [
    'imagenette_full_page1.png',
    'imagewoof_full_page1.png',
    'imageyellow_full_page1.png',
    'imagefruit_full_page1.png',
    'imagesquawk_full_page1.png',
    'imagemeow_full_page1.png'
  ];
  let idx = 0;
  let timer = null;
  const imgEl = document.getElementById('carousel-img');
  const nameEl = document.getElementById('carousel-filename');
  const prevBtn = document.getElementById('carousel-prev');
  const nextBtn = document.getElementById('carousel-next');
  function show(idx_) {
    imgEl.src = carouselDir + imgFiles[idx_];
    nameEl.textContent = imgFiles[idx_];
  }
  prevBtn.onclick = function() {
    idx = (idx - 1 + imgFiles.length) % imgFiles.length;
    show(idx);
    resetTimer();
  };
  nextBtn.onclick = function() {
    idx = (idx + 1) % imgFiles.length;
    show(idx);
    resetTimer();
  };
  function autoPlay() {
    timer = setInterval(() => {
      idx = (idx + 1) % imgFiles.length;
      show(idx);
    }, 2500);
  }
  function resetTimer() {
    if (timer) clearInterval(timer);
    autoPlay();
  }
  if(imgFiles.length) {
    show(0);
    autoPlay();
  }
});
