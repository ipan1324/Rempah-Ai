// ============================================================
// script.js – RempahAI Global JavaScript
// ============================================================

// ─── Navbar Scroll Effect ─────────────────────────────────────
(function () {
  'use strict';

  const navbar = document.getElementById('mainNavbar');

  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 50) {
        navbar.style.background = 'rgba(13, 14, 26, 0.97)';
        navbar.style.boxShadow  = '0 4px 20px rgba(0,0,0,0.4)';
      } else {
        navbar.style.background = 'rgba(13, 14, 26, 0.85)';
        navbar.style.boxShadow  = 'none';
      }
    }, { passive: true });
  }

  // ─── Smooth Scroll untuk anchor links ──────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;

      const target = document.querySelector(targetId);
      if (target) {
        e.preventDefault();
        const navHeight = navbar ? navbar.offsetHeight : 0;
        const targetPos = target.getBoundingClientRect().top + window.scrollY - navHeight - 16;
        window.scrollTo({ top: targetPos, behavior: 'smooth' });
      }
    });
  });

  // ─── Animasi Confidence & Probability Bar ──────────────────
  // Trigger animasi CSS bar dengan IntersectionObserver
  const bars = document.querySelectorAll('.confidence-bar-fill, .prob-bar');

  if (bars.length > 0) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el = entry.target;
          // Ambil target width dari CSS custom property
          const targetWidth =
            el.style.getPropertyValue('--target-width') ||
            el.style.getPropertyValue('--prob-width') ||
            '0%';
          // Terapkan animasi
          requestAnimationFrame(() => {
            el.style.width = targetWidth;
          });
          observer.unobserve(el);
        }
      });
    }, { threshold: 0.1 });

    bars.forEach(bar => observer.observe(bar));
  }

  // ─── Auto-dismiss Alert ─────────────────────────────────────
  // Alert otomatis menghilang setelah 5 detik
  document.querySelectorAll('.alert.alert-dismissible').forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) bsAlert.close();
    }, 5000);
  });

  // ─── Tooltip Bootstrap ─────────────────────────────────────
  // Aktifkan semua tooltip Bootstrap 5
  const tooltipTriggers = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltipTriggers.forEach(el => new bootstrap.Tooltip(el));

})();
