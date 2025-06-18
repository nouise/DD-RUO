// 页面加载动画
document.addEventListener('DOMContentLoaded', function() {
    // 添加页面加载动画
    const sections = document.querySelectorAll('.section');
    
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
});

// 平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// 复制BibTeX功能
function copyBibTeX() {
    const bibtexText = document.querySelector('.bibtex code').textContent;
    navigator.clipboard.writeText(bibtexText).then(function() {
        // 显示复制成功提示
        const button = event.target;
        const originalText = button.textContent;
        button.textContent = '已复制!';
        button.style.backgroundColor = '#10b981';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.backgroundColor = '';
        }, 2000);
    });
}

// 添加复制按钮到BibTeX区域
document.addEventListener('DOMContentLoaded', function() {
    const bibtexSection = document.querySelector('.bibtex');
    if (bibtexSection) {
        const copyButton = document.createElement('button');
        copyButton.textContent = '复制引用';
        copyButton.className = 'copy-button';
        copyButton.onclick = copyBibTeX;
        
        // 添加复制按钮样式
        const style = document.createElement('style');
        style.textContent = `
            .bibtex {
                position: relative;
            }
            .copy-button {
                position: absolute;
                top: 15px;
                right: 15px;
                background-color: #2563eb;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: background-color 0.3s ease;
            }
            .copy-button:hover {
                background-color: #1d4ed8;
            }
        `;
        document.head.appendChild(style);
        
        bibtexSection.appendChild(copyButton);
    }
});

// 表格行高亮效果
document.addEventListener('DOMContentLoaded', function() {
    const tableRows = document.querySelectorAll('tbody tr');
    tableRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#eff6ff';
        });
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    });
});

// 链接按钮点击效果
document.querySelectorAll('.link-button').forEach(button => {
    button.addEventListener('click', function(e) {
        // 如果链接为空（#），阻止默认行为并显示提示
        if (this.getAttribute('href') === '#') {
            e.preventDefault();
            alert('请在实际使用时替换为真实链接');
        }
    });
});

// 性能亮点数字动画
function animateNumbers() {
    const highlights = document.querySelectorAll('.highlight-number');
    
    highlights.forEach(highlight => {
        const finalValue = highlight.textContent;
        const isPercentage = finalValue.includes('%');
        const isMultiplier = finalValue.includes('x');
        
        let numericValue = parseFloat(finalValue.replace(/[^\d.]/g, ''));
        let currentValue = 0;
        const increment = numericValue / 50; // 50步动画
        
        const timer = setInterval(() => {
            currentValue += increment;
            if (currentValue >= numericValue) {
                currentValue = numericValue;
                clearInterval(timer);
            }
            
            let displayValue = currentValue.toFixed(1);
            if (isPercentage) {
                displayValue += '%';
            } else if (isMultiplier) {
                displayValue += 'x';
            }
            
            highlight.textContent = displayValue;
        }, 30);
    });
}

// 当性能亮点区域进入视口时开始动画
const highlightObserver = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateNumbers();
            highlightObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

document.addEventListener('DOMContentLoaded', function() {
    const highlightGrid = document.querySelector('.highlight-grid');
    if (highlightGrid) {
        highlightObserver.observe(highlightGrid);
    }
});

