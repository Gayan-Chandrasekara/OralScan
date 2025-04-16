const CACHE_NAME = 'oral-cancer-app-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/overlay.js',
  '/service-worker.js',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
  'https://firebasestorage.googleapis.com/v0/b/oral-cancer-detector-1e004.firebasestorage.app/o/model.onnx?alt=media&token=d17886be-78e8-4916-a377-4ff3c6b7ef68'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.map(key => key !== CACHE_NAME && caches.delete(key)))
    )
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(resp => resp || fetch(event.request))
  );
});
