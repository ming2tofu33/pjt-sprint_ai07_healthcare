(function () {
  'use strict';

  var MIN_SLIDE = 1;
  var MAX_SLIDE = 18;
  var slideMatch = window.location.pathname.match(/slide_(\d{2})\.html$/i);

  if (!slideMatch) {
    console.warn('[presentation-nav] slide_XX.html 파일명이 아니어서 내비게이션을 비활성화합니다.');
    return;
  }

  var currentSlide = parseInt(slideMatch[1], 10);
  if (!Number.isFinite(currentSlide)) {
    console.warn('[presentation-nav] 현재 슬라이드 번호를 파싱하지 못했습니다.');
    return;
  }
  var THEME_ICON_MAP = {
    1: null,
    2: 'fa-users',
    3: 'fa-clipboard-list',
    4: 'fa-lightbulb',
    5: 'fa-random',
    6: 'fa-file-code',
    7: 'fa-exclamation-circle',
    8: 'fa-tools',
    9: 'fa-chart-bar',
    10: 'fa-cogs',
    11: 'fa-balance-scale-right',
    12: 'fa-flask',
    13: 'fa-lightbulb',
    14: 'fa-folder-tree',
    15: 'fa-layer-group',
    16: 'fa-trophy',
    17: 'fa-clipboard-check',
    18: 'fa-comments'
  };

  function pad2(n) {
    return String(n).padStart(2, '0');
  }

  function renderPageBadge() {
    if (!document.body) {
      return;
    }

    var badge = document.createElement('div');
    badge.id = 'presentation-page-badge';
    badge.textContent = String(currentSlide);
    badge.setAttribute('aria-label', '현재 슬라이드 ' + currentSlide + '페이지');
    badge.style.position = 'fixed';
    badge.style.top = '14px';
    badge.style.right = '14px';
    badge.style.width = '34px';
    badge.style.height = '34px';
    badge.style.borderRadius = '9999px';
    badge.style.display = 'flex';
    badge.style.alignItems = 'center';
    badge.style.justifyContent = 'center';
    badge.style.fontFamily = "'Fira Code', 'Noto Sans KR', monospace";
    badge.style.fontWeight = '700';
    badge.style.fontSize = '14px';
    badge.style.color = '#004346';
    badge.style.background = 'rgba(255, 255, 255, 0.92)';
    badge.style.border = '1px solid #75DDDD';
    badge.style.boxShadow = '0 2px 6px rgba(23, 42, 58, 0.16)';
    badge.style.zIndex = '9999';
    badge.style.pointerEvents = 'none';

    document.body.appendChild(badge);
  }

  function ensureFontAwesome() {
    var hasFontAwesome = !!document.querySelector('link[href*="font-awesome"], link[href*="fontawesome"]');
    if (hasFontAwesome || !document.head) {
      return;
    }

    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';
    document.head.appendChild(link);
  }

  function renderThemeIcon() {
    if (!document.body) {
      return;
    }

    ensureFontAwesome();

    var existing = document.getElementById('presentation-theme-icon');
    if (existing) {
      existing.remove();
    }

    var iconClass = THEME_ICON_MAP[currentSlide];
    if (!iconClass) {
      return;
    }
    var iconWrap = document.createElement('div');
    iconWrap.id = 'presentation-theme-icon';
    iconWrap.setAttribute('aria-hidden', 'true');
    iconWrap.style.position = 'absolute';
    iconWrap.style.top = '0';
    iconWrap.style.right = '0';
    iconWrap.style.padding = '40px';
    iconWrap.style.color = '#9CA3AF';
    iconWrap.style.opacity = '0.08';
    iconWrap.style.fontSize = '9rem';
    iconWrap.style.lineHeight = '1';
    iconWrap.style.pointerEvents = 'none';
    iconWrap.style.zIndex = '0';

    var icon = document.createElement('i');
    icon.className = 'fas ' + iconClass;
    iconWrap.appendChild(icon);

    document.body.appendChild(iconWrap);
  }

  function goToSlide(nextSlide) {
    if (nextSlide < MIN_SLIDE || nextSlide > MAX_SLIDE) {
      return;
    }
    if (nextSlide === currentSlide) {
      return;
    }
    window.location.href = 'slide_' + pad2(nextSlide) + '.html';
  }

  function isEditableTarget(target) {
    if (!target) {
      return false;
    }
    var tag = (target.tagName || '').toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable;
  }

  function getFullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement || null;
  }

  function requestFullscreen() {
    var el = document.documentElement;
    var fn = el.requestFullscreen || el.webkitRequestFullscreen || el.msRequestFullscreen;
    if (!fn) {
      return;
    }

    try {
      var ret = fn.call(el);
      if (ret && typeof ret.catch === 'function') {
        ret.catch(function () {
          // 브라우저 정책으로 거절되어도 슬라이드 이동은 계속 허용
        });
      }
    } catch (_) {
      // 브라우저별 예외는 무시
    }
  }

  function exitFullscreen() {
    var fn = document.exitFullscreen || document.webkitExitFullscreen || document.msExitFullscreen;
    if (!fn) {
      return;
    }

    try {
      var ret = fn.call(document);
      if (ret && typeof ret.catch === 'function') {
        ret.catch(function () {
          // ignore
        });
      }
    } catch (_) {
      // ignore
    }
  }

  var autoFullscreenAttempted = false;

  function requestFullscreenOnce() {
    if (autoFullscreenAttempted) {
      return;
    }
    autoFullscreenAttempted = true;
    requestFullscreen();
  }

  renderThemeIcon();
  renderPageBadge();

  document.addEventListener('keydown', function (event) {
    if (isEditableTarget(event.target)) {
      return;
    }

    var key = event.key;
    var isNext = key === ' ' || key === 'Spacebar' || key === 'ArrowRight' || key === 'ArrowDown';
    var isPrev = key === 'ArrowLeft' || key === 'ArrowUp' || key === 'Backspace';
    var isFullscreenToggle = key === 'f' || key === 'F';

    if (!isNext && !isPrev && !isFullscreenToggle) {
      return;
    }

    event.preventDefault();

    if (isFullscreenToggle) {
      if (getFullscreenElement()) {
        exitFullscreen();
      } else {
        requestFullscreen();
      }
      return;
    }

    requestFullscreenOnce();

    if (isNext) {
      goToSlide(currentSlide + 1);
      return;
    }

    if (isPrev) {
      goToSlide(currentSlide - 1);
    }
  });
})();
