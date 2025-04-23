const buttons = document.querySelectorAll(".tab-button");
const contents = document.querySelectorAll(".tab-content");
const modelSelect = document.getElementById("modelsSelect");

buttons.forEach(button => {
    button.addEventListener("click", () => {
        buttons.forEach(btn => btn.classList.remove("active"));
        contents.forEach(content => content.classList.remove("active"));

        button.classList.add("active");
        const tabId = button.getAttribute("data-tab");
        document.getElementById(tabId).classList.add("active");
    });
});

modelSelect.addEventListener("change", function() {
    document.getElementById("selectModel").submit();
});
