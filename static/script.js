// JavaScript function to format numbers
function formatNumber(value) {
    value = parseFloat(value);
    if (value >= 1000) {
        return (value / 1000).toFixed(1).replace('.0', '') + 'k';
    } else {
        return value.toFixed(0); // Đảm bảo không có số lẻ khi số < 1000
    }
}

document.addEventListener('DOMContentLoaded', function () {
    // Format review counts
    document.querySelectorAll('.rating').forEach(function (element) {
        var reviewCount = element.getAttribute('data-review-count') || 0;  // Set default to 0 if no value
        element.querySelector('.formatted-review-count').innerText = formatNumber(reviewCount);
    });

    // Format quantity sold
    document.querySelectorAll('.sold-count').forEach(function (element) {
        var quantitySold = element.getAttribute('data-quantity-sold') || 0;  // Set default to 0 if no value
        element.querySelector('.formatted-quantity-sold').innerText = formatNumber(quantitySold);
    });

    // Get all 'Add to Cart' forms
    const addToCartForms = document.querySelectorAll('.add-to-cart-form');

    addToCartForms.forEach(function (form) {
        form.addEventListener('click', function (e) {
            const productId = this.getAttribute('data-product-id');

            // Send AJAX request to add product to cart
            fetch('/add-to-cart', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrf_token')  // If using CSRF tokens
                },
                body: JSON.stringify({
                    product_id: productId,
                    quantity: 1  // Set default quantity to 1
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update cart count or show a success message
                    updateCartCount();  // Call function to update cart count
                } else {
                    alert('Có lỗi xảy ra khi thêm vào giỏ hàng');
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });

            e.preventDefault();  // Prevent the default form submission
        });
    });

    // Function to update the cart count in the DOM
    function updateCartCount() {
        const cartCountElement = document.querySelector('.cart-count');
        let cartCount = parseInt(cartCountElement.innerText) || 0;  // Get the current count or default to 0
        cartCount += 1;  // Increment the cart count
        cartCountElement.innerText = cartCount;  // Update the cart count in the DOM
    }
});


// Function to get CSRF token from cookies (if using Flask-WTF for CSRF protection)
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}


// Function to create product card HTML
function createProductCard(product) {
    return `
    <div class="product-card">
        <div class="product-image">
            <img src="${product.images.split(',')[0].replace('[', '').replace(']', '').replace("'", '').trim()}" alt="${product.name}">
            ${product.discount > 0 ? `<span class="discount-label">${((product.discount / product.list_price) * 100).toFixed(0)}% Off</span>` : ''}
        </div>
        <div class="product-info">
            <h3>${product.name}</h3>
            <p class="price">${(product.list_price - product.discount).toLocaleString('en').replace(',', '.')} VND</p>
            <div class="product-rating">
                <p class="rating" data-review-count="${product.review_count}">
                    ${product.average_rating} (<span class="formatted-review-count">${formatNumber(product.review_count)}</span> reviews)
                </p>
                <p class="sold-count" data-quantity-sold="${product.quantity_sold}">
                    Đã bán <span class="formatted-quantity-sold">${formatNumber(product.quantity_sold)}</span>
                </p>
            </div>
            <a href="/product/${product.id}" class="details-btn">View Details</a>
        </div>
    </div>`;
}

// Pagination variables
let currentPage = 1;
const viewMoreBtn = document.getElementById('view-more-btn');

// Add event listener to "View More" button
viewMoreBtn.addEventListener('click', function() {
    currentPage += 1;  // Increase the page number

    // Fetch more products using AJAX
    fetch(`/load-more-top-rated?page=${currentPage}`)
        .then(response => response.json())
        .then(data => {
            const productGrid = document.getElementById('product-grid');

            // Loop through the fetched products and append them to the product grid
            data.forEach(product => {
                const productCard = createProductCard(product);
                productGrid.innerHTML += productCard;
            });

            // Disable the button if no more products are left
            if (data.length < 8) {
                viewMoreBtn.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error fetching more products:', error);
        });
});

// Format review counts

document.querySelectorAll('.rating').forEach(function(element) {
    var reviewCount = element.getAttribute('data-review-count') || 0;  // Set default to 0 if no value
    element.querySelector('.formatted-review-count').innerText = formatNumber(reviewCount);
});

// Format quantity sold
document.querySelectorAll('.sold-count').forEach(function(element) {
    var quantitySold = element.getAttribute('data-quantity-sold') || 0;  // Set default to 0 if no value
    element.querySelector('.formatted-quantity-sold').innerText = formatNumber(quantitySold);
});

document.addEventListener('DOMContentLoaded', function () {
    // Handle 'Add to Cart' functionality
    const addToCartForms = document.querySelectorAll('.add-to-cart-form');

    addToCartForms.forEach(function (form) {
        form.addEventListener('submit', function (e) {
            e.preventDefault();
            const productId = this.getAttribute('data-product-id');

            // Send an AJAX request to add the product to the cart
            fetch('/add-to-cart', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrf_token')  // CSRF token if applicable
                },
                body: JSON.stringify({
                    product_id: productId,
                    quantity: 1
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Sản phẩm đã được thêm vào giỏ hàng!');
                } else {
                    alert('Có lỗi xảy ra khi thêm vào giỏ hàng');
                }
            })
            .catch(error => console.error('Error:', error));
        });
    });
});
