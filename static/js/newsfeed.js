async function fetchArticlesAsync(userId) {
  const response = await fetch(`/articles/${userId}`);
  if (!response.ok) {
    throw new Error(`Response status: ${response.status}`);
  }
  const result = await response.json();
  console.log(result);
  return result;
}

async function fetchRelatedArticlesAsync(embeddingId) {
  const response = await fetch(`/related/${embeddingId}`);
  if (!response.ok) {
    throw new Error(`Response status: ${response.status}`);
  }
  const result = await response.json();
  console.log(result);
  return result;
}
