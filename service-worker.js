// service-worker.js
const CACHE_NAME = 'oral-cancer-detector-cache-v1';
const urlsToCache = [
    '/',
    '/index.html',
    '/overlay.js', 
    '/style.css',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.7.0/dist/onnxruntime-web.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs',
    // Add other assets you want to cache (e.g., model files, images)
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                return cache.addAll(urlsToCache);
            })
    );
});

self.addEventListener('activate', (event) => {
    const cacheWhitelist = [CACHE_NAME];
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (!cacheWhitelist.includes(cacheName)) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((cachedResponse) => {
            if (cachedResponse) {
                return cachedResponse; // Return the cached resource if available
            }
            return fetch(event.request); // Otherwise, fetch it from the network
        })
    );
});
