(function () {
  var videos = Array.prototype.slice.call(document.querySelectorAll('.video-card video'));

  function markMissing(video) {
    var card = video.closest('.video-card');
    if (card) card.classList.add('is-missing');
  }

  function play(video) {
    video.muted = true;
    var p = video.play();
    if (p && p.catch) p.catch(function () {});
  }

  videos.forEach(function (video) {
    video.addEventListener('error', function () { markMissing(video); });
    video.addEventListener('loadedmetadata', function () {
      if (video.videoWidth && video.videoHeight) {
        video.style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight;
      }
    });
    if (video.error || video.networkState === HTMLMediaElement.NETWORK_NO_SOURCE) {
      markMissing(video);
    } else {
      play(video);
    }
  });

  // browsers that block autoplay (e.g. low-power mode) start the loops on the first interaction
  function kick() {
    videos.forEach(function (video) { if (video.paused) play(video); });
    document.removeEventListener('touchstart', kick);
    document.removeEventListener('click', kick);
  }
  document.addEventListener('touchstart', kick, { passive: true });
  document.addEventListener('click', kick);

  Array.prototype.forEach.call(document.querySelectorAll('.copy-button'), function (button) {
    button.addEventListener('click', function () {
      var target = document.getElementById(button.getAttribute('data-copy-target'));
      if (!target || !navigator.clipboard) return;
      navigator.clipboard.writeText(target.textContent).then(function () {
        var label = button.textContent;
        button.textContent = 'Copied';
        setTimeout(function () { button.textContent = label; }, 1500);
      });
    });
  });
})();
