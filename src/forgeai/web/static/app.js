// Show server errors from HTMX requests in the page instead of failing silently.
document.addEventListener("htmx:responseError", function (event) {
  var target = document.getElementById("error");
  if (!target) return;
  var text = event.detail.xhr.responseText || "Something went wrong.";
  try { text = JSON.parse(text).detail || text; } catch (e) { /* plain text */ }
  target.textContent = text;
  target.hidden = false;
});
document.addEventListener("htmx:beforeRequest", function () {
  var target = document.getElementById("error");
  if (target) target.hidden = true;
});
// Clear and refocus the chat box after a question is sent.
document.addEventListener("htmx:afterRequest", function (event) {
  var form = event.target;
  if (form && form.matches && form.matches("form[data-reset]") && event.detail.successful) {
    form.reset();
    var box = form.querySelector("textarea, input[type=text]");
    if (box) box.focus();
    var log = document.getElementById("transcript");
    if (log) log.lastElementChild && log.lastElementChild.scrollIntoView({ block: "start" });
  }
});
