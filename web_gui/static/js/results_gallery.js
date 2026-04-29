window.ResultGallery = {
    setMessage(containerId, message, kind = 'info') {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = `<div class="viz-empty viz-empty-${kind}">${message}</div>`;
    },

    setLoading(containerId, message = 'Loading visual results…') {
        this.setMessage(containerId, message, 'loading');
    },

    renderImages(containerId, images, emptyMessage = 'No visual results yet.') {
        const container = document.getElementById(containerId);
        if (!container) return;
        if (!images || !images.length) {
            this.setMessage(containerId, emptyMessage, 'empty');
            return;
        }

        container.innerHTML = images.map((image) => `
            <article class="viz-card">
                <div class="viz-card-header">
                    <h4>${image.title || image.name}</h4>
                </div>
                <a href="${image.url}" target="_blank" rel="noopener noreferrer">
                    <img src="${image.url}" alt="${image.title || image.name}" loading="lazy">
                </a>
                <div class="viz-card-meta">${image.name}</div>
            </article>
        `).join('');
    },

    async fetchAndRender(group, containerId, options = {}) {
        const {
            loadingMessage = 'Loading visual results…',
            emptyMessage = 'No visual results yet.',
            onSuccess,
            onError,
        } = options;

        this.setLoading(containerId, loadingMessage);
        try {
            const res = await fetch(`/api/visualizations/${group}`);
            const data = await res.json();
            if (!res.ok || data.error) {
                throw new Error(data.error || `Failed to load ${group} visuals.`);
            }
            this.renderImages(containerId, data.images, emptyMessage);
            if (onSuccess) onSuccess(data.images || []);
        } catch (error) {
            this.setMessage(containerId, `Unable to load visuals: ${error.message}`, 'error');
            if (onError) onError(error);
        }
    },
};
