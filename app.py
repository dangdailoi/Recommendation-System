# Standard library imports
import os
import csv
import threading
import time
from datetime import datetime

# Third-party imports
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    jsonify, flash
)
from flask_sqlalchemy import SQLAlchemy

# Local application imports
from config import Config
from db.dbo import (
    Product, Tracking, User, Category, ProductCategory,
    UserActivityLog, db
)
from search_engine import ImageSearch, TextSearch
from recommendation_system import DeepContentBasedFiltering, DeepQNetwork

# Create Flask application
app = Flask(__name__)
app.config.from_object(Config)

# Initialize SQLAlchemy
db.init_app(app)

# Constants from config
MODEL_PATH = app.config['MODEL_PATH']
CSV_FILE_PATH = app.config['CSV_FILE_PATH']
MODEL_UPDATE_INTERVAL = app.config['MODEL_UPDATE_INTERVAL']

# Ensure CSV file exists and has headers
def ensure_csv_file(csv_file_path):
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['user_id', 'product_id', 'activity_type', 'timestamp'])

ensure_csv_file(CSV_FILE_PATH)

# Activity priority mapping
ACTIVITY_PRIORITY = {'select': 4, 'favourite': 3, 'view': 2, 'search': 1}

# Initialize search and recommendation engines
image_search_engine = ImageSearch(config_file_path='model.json')
text_search_engine = TextSearch()
content_based_filter = DeepContentBasedFiltering('model.json')

# Initialize DQN agent
STATE_SIZE = 10  # State size, can be adjusted
ACTION_SIZE = 50  # Assume 50 actions (e.g., popular products)
dqn_agent = DeepQNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE)

# Load DQN model if it exists
dqn_agent.load(MODEL_PATH)

# Helper Functions
def extract_state_from_db(user_id):
    logs = UserActivityLog.query.filter_by(user_id=user_id).order_by(
        UserActivityLog.view_end_time.desc()
    ).limit(STATE_SIZE).all()
    state = [log.product_id for log in logs]
    state += [0] * (STATE_SIZE - len(state))  # Padding if needed
    return np.array(state)

def log_to_csv(user_id, product_id, activity_type):
    with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, product_id, activity_type, datetime.now()])

def log_user_activity(user_id, product_id, activity_type, quantity=1, view_end_time=None):
    valid_activity_types = ['view', 'select', 'purchase', 'remove_from_cart', 'favourite', 'search']
    if activity_type not in valid_activity_types:
        raise ValueError(f"Invalid activity type: {activity_type}")

    activity_log = UserActivityLog(
        user_id=user_id,
        product_id=product_id,
        activity_type=activity_type,
        quantity=quantity,
        view_end_time=view_end_time
    )
    db.session.add(activity_log)
    db.session.commit()

def is_favourited(user_id, product_id):
    if user_id:
        favourite_entry = UserActivityLog.query.filter_by(
            user_id=user_id, product_id=product_id, activity_type='favourite'
        ).first()
        return bool(favourite_entry)
    return False

def get_activity_score(activity_type):
    score_mapping = {
        'view': 1,
        'favourite': 3,
        'select': 2,
        'remove_from_cart': -1,
        'search': 0
    }
    return score_mapping.get(activity_type, 0)

def categorize_product(category_id):
    while category_id:
        category = Category.query.get(category_id)
        if not category:
            break
        if category.parent_category_id == 915:
            return 'fashion'
        elif category.parent_category_id == 316:
            return 'book'
        category_id = category.parent_category_id
    return 'other'

# Background Tasks
def update_dqn_model():
    with app.app_context():
        while True:
            print("Starting DQN training at regular intervals...")

            # Get list of user_ids from the database
            users = User.query.all()

            for user in users:
                user_id = user.user_id

                # Extract the current state for the user
                state = extract_state_from_db(user_id).reshape(1, -1)

                # Get the latest activity log for the user
                latest_log = UserActivityLog.query.filter_by(user_id=user_id).order_by(
                    UserActivityLog.view_end_time.desc()
                ).first()

                if latest_log:
                    activity_type = latest_log.activity_type
                    reward = get_activity_score(activity_type)
                else:
                    # Assign a default reward if no activity is found
                    reward = 0

                # Agent selects an action based on the current state
                action = dqn_agent.act(state)

                # Simulate the next state
                next_state = extract_state_from_db(user_id).reshape(1, -1)

                done = False  # Define end condition if necessary

                # Train the DQN model
                dqn_agent.remember(state, action, reward, next_state, done)
                dqn_agent.replay()

            # Save the model after training
            dqn_agent.save(MODEL_PATH)
            print(f"DQN model has been updated and saved at {MODEL_PATH}")

            # Wait for the specified interval before the next training session
            time.sleep(MODEL_UPDATE_INTERVAL)

# Start background thread to update model
threading.Thread(target=update_dqn_model, daemon=True).start()

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Query the database for the user by email and password
        user = User.query.filter_by(email=email, password=password).first()

        if user:
            session['user_id'] = user.user_id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout_user():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login_user'))

# Home Page
@app.route('/')
def index():
    return redirect(url_for('home'))

def get_recommendations(user_id, limit=None):
    # Fetch activity logs from CSV
    activity_logs = []

    if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
        with open(CSV_FILE_PATH, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                if row[0] == str(user_id):
                    activity_logs.append({
                        'product_id': int(row[1]),
                        'activity_type': row[2],
                        'datetime': row[3]
                    })

    if not activity_logs:
        return []

    # Sort activity_logs by priority and time
    activity_logs.sort(
        key=lambda log: (ACTIVITY_PRIORITY.get(log['activity_type'], 99), log['datetime']),
        reverse=True
    )

    # Get product_ids from activities
    interacted_product_ids = [log['product_id'] for log in activity_logs]

    # DQN Recommendations
    dqn_recommended_ids = []
    state = extract_state_from_db(user_id).reshape(1, -1)
    action = dqn_agent.act(state)
    if interacted_product_ids:
        selected_product_id = interacted_product_ids[action % len(interacted_product_ids)]
        dqn_recommended_ids.append(selected_product_id)

    # Get similar products
    similar_product_ids = []
    for product_id in interacted_product_ids:
        product_category = ProductCategory.query.filter_by(product_id=product_id).first()
        if product_category:
            category_type = categorize_product(product_category.category_id)
            similar_ids = content_based_filter.recommend(
                product_id=product_id,
                product_type=category_type,
                top_k=5,
                exclude_viewed=False
            )
            similar_product_ids.extend(similar_ids)

    # Remove duplicates and limit number of products
    all_recommended_ids = list(dict.fromkeys(
        interacted_product_ids + dqn_recommended_ids + similar_product_ids
    ))
    if limit:
        all_recommended_ids = all_recommended_ids[:limit]

    # Query products from the database
    recommended_products_with_tracking = (
        db.session.query(Product, Tracking)
        .join(Tracking, Tracking.product_id == Product.product_id)
        .filter(Product.product_id.in_(all_recommended_ids))
        .all()
    )

    # Sort products according to order in all_recommended_ids
    product_order_map = {pid: idx for idx, pid in enumerate(all_recommended_ids)}
    recommended_products_with_tracking.sort(
        key=lambda x: product_order_map.get(x[0].product_id, len(all_recommended_ids))
    )

    return recommended_products_with_tracking

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))

    user_id = session.get('user_id')

    # Get recommended products
    recommended_products_with_tracking = get_recommendations(user_id, limit=16)

    # Get top-rated products
    top_rated_products = (
        db.session.query(Product, Tracking)
        .join(Tracking)
        .order_by(Tracking.rating_average.desc(), Tracking.review_count.desc())
        .limit(24)
        .all()
    )

    # Favourite status
    favourite_status_recommended = {
        product.Product.product_id: is_favourited(user_id, product.Product.product_id)
        for product in recommended_products_with_tracking
    }
    favourite_status_top_rated = {
        product.Product.product_id: is_favourited(user_id, product.Product.product_id)
        for product in top_rated_products
    }

    cart_count = len(session.get('cart', []))
    search_history = session.get('search_history', [])

    return render_template(
        'home.html',
        recommended_products=recommended_products_with_tracking,
        top_rated_products=top_rated_products,
        favourite_status_recommended=favourite_status_recommended,
        favourite_status_top_rated=favourite_status_top_rated,
        cart_count=cart_count,
        search_history=search_history
    )

# Product Browsing Routes
@app.route('/top-rated/<int:page>')
def show_top_rated(page=1):
    user_id = session.get('user_id')
    per_page = 48
    start = (page - 1) * per_page

    total_products = db.session.query(Tracking).count()
    total_pages = (total_products + per_page - 1) // per_page

    top_rated_products = (
        db.session.query(Product, Tracking)
        .join(Tracking)
        .order_by(Tracking.rating_average.desc(), Tracking.review_count.desc())
        .offset(start).limit(per_page).all()
    )

    favourite_status_top_rated = {
        product.Product.product_id: is_favourited(user_id, product.Product.product_id)
        for product in top_rated_products
    }

    cart_count = len(session.get('cart', []))
    search_history = session.get('search_history', [])
    session['search_history'] = list(dict.fromkeys(search_history))

    start_page = max(1, page - 2)
    end_page = min(total_pages, page + 2)

    return render_template(
        'top_rated.html',
        products=top_rated_products,
        favourite_status=favourite_status_top_rated,
        cart_count=cart_count,
        current_page=page,
        total_pages=total_pages,
        start_page=start_page,
        end_page=end_page,
        search_history=session['search_history']
    )

# Product Details Routes
@app.route('/product/<int:product_id>', methods=['GET', 'POST'])
def view_product(product_id):
    # Log 'view' activity when the product page is viewed
    user_id = session.get('user_id')
    if user_id:
        log_user_activity(user_id, product_id, 'view')
        log_to_csv(user_id, product_id, 'view')

    product = Product.query.get_or_404(product_id)
    tracking = Tracking.query.filter_by(product_id=product_id).first_or_404()
    favourite_status = is_favourited(user_id, product.product_id)

    # Get similar products
    product_category = ProductCategory.query.filter_by(product_id=product_id).first()
    category_type = categorize_product(product_category.category_id)
    similar_ids = content_based_filter.recommend(
        product_id=product_id,
        product_type=category_type,
        top_k=9,
        exclude_viewed=True,
        viewed_product_ids=None,
        content_based=True
    )
    similar_products = Product.query.filter(
        Product.product_id.in_(similar_ids)
    ).all()
    tracking_data = {
        tracking.product_id: tracking for tracking in
        Tracking.query.filter(
            Tracking.product_id.in_([p.product_id for p in similar_products])
        ).all()
    }
    products_with_tracking = [
        (product, tracking_data.get(product.product_id)) for product in similar_products
    ]
    favourite_status_similar = {
        product.product_id: is_favourited(user_id, product.product_id)
        for product, tracking in products_with_tracking
    }

    highlight = (
        product.product_highlights.split(',')
        if isinstance(product.product_highlights, str)
        else product.product_highlights
    )
    cart_count = len(session.get('cart', []))

    if request.method == 'POST':
        quantity = int(request.form.get('quantity', 1))
        price = float(product.product_price - tracking.discount)
        cart_item = {
            'product_id': product.product_id,
            'product_name': product.product_name,
            'price': price,
            'quantity': quantity
        }

        cart = session.setdefault('cart', [])
        existing_item = next(
            (item for item in cart if item['product_id'] == product.product_id),
            None
        )

        if existing_item:
            existing_item['quantity'] += quantity
        else:
            cart.append(cart_item)

        session.modified = True

        # Log 'select' activity when product is added to cart
        if user_id:
            log_user_activity(user_id, product_id, 'select', quantity=quantity)
            log_to_csv(user_id, product_id, 'select')

        return redirect(url_for('view_cart'))

    return render_template(
        'products.html',
        cart_count=cart_count,
        product=product,
        favourite_status=favourite_status,
        tracking=tracking,
        highlight=highlight,
        similar_products=products_with_tracking,
        favourite_status_similar=favourite_status_similar
    )

# Cart and Checkout Routes
@app.route('/add-to-cart', methods=['POST'])
def add_product_to_cart():
    data = request.get_json()
    product_id = data.get('product_id')
    quantity = int(data.get('quantity', 1))

    product = Product.query.get_or_404(product_id)
    tracking = Tracking.query.filter_by(product_id=product_id).first()

    price = float(product.product_price - tracking.discount)
    cart_item = {
        'product_id': product.product_id,
        'product_name': product.product_name,
        'price': price,
        'quantity': quantity
    }

    cart = session.setdefault('cart', [])
    existing_item = next(
        (item for item in cart if item['product_id'] == product.product_id),
        None
    )

    if existing_item:
        existing_item['quantity'] += quantity
    else:
        cart.append(cart_item)

    session.modified = True

    # Log 'select' activity when product is added to cart
    user_id = session.get('user_id')
    if user_id:
        log_user_activity(user_id, product_id, 'select', quantity=quantity)
        log_to_csv(user_id, product_id, 'select')

    return jsonify({'success': True})

@app.route('/cart')
def view_cart():
    cart_items = session.get('cart', [])
    grouped_cart_items = {}

    for item in cart_items:
        key = str(item['product_id'])
        if key in grouped_cart_items:
            grouped_cart_items[key]['quantity'] += item['quantity']
        else:
            grouped_cart_items[key] = item

    total = sum(
        float(item['price']) * item['quantity'] for item in grouped_cart_items.values()
    )

    return render_template('cart.html', cart_items=grouped_cart_items.values(), total=total)

@app.route('/remove-from-cart', methods=['POST'])
def remove_product_from_cart():
    product_id = request.form.get('product_id')

    if not product_id:
        return redirect(url_for('view_cart'))

    # Get the cart from the session
    cart = session.get('cart', [])
    cart = [item for item in cart if str(item['product_id']) != str(product_id)]

    # Update the cart in the session
    session['cart'] = cart
    session.modified = True

    # Log 'remove_from_cart' activity
    user_id = session.get('user_id')
    if user_id:
        log_user_activity(user_id, product_id, 'remove_from_cart')
        log_to_csv(user_id, product_id, 'remove_from_cart')

    return redirect(url_for('view_cart'))

@app.route('/checkout', methods=['GET', 'POST'])
def checkout():
    if request.method == 'POST':
        # Process payment or add product to order
        return redirect("https://dailoi-ddl.glitch.me/")
    else:
        return redirect("https://dailoi-ddl.glitch.me/")

# Search Routes
@app.route('/search-by-text')
def search_by_text():
    query_str = request.args.get('query', '')
    page = request.args.get('page', 1, type=int)
    per_page = 24

    if query_str:
        # Get search history from session and manage duplicates
        search_history = session.setdefault('search_history', [])
        if query_str not in search_history:
            search_history.insert(0, query_str)
        session['search_history'] = search_history[:5]  # Keep only the 5 most recent terms

        # Proceed with search logic
        results = text_search_engine.search(query_str)

        formatted_results = []
        for product_id, _, _, _, _ in results:
            product = Product.query.filter_by(product_id=product_id).first()
            tracking = Tracking.query.filter_by(product_id=product_id).first()
            if product and tracking:
                formatted_results.append((product, tracking))

        # Log the first result if user is logged in
        if 'user_id' in session:
            user_id = session.get('user_id')
            for product, _ in formatted_results[:1]:
                log_to_csv(user_id, product.product_id, 'search')
                log_user_activity(user_id, product.product_id, 'search', quantity=1)

        total_products = len(formatted_results)
        total_pages = (total_products + per_page - 1) // per_page
        paginated_results = formatted_results[(page - 1) * per_page: page * per_page]
    else:
        paginated_results = []
        total_pages = 0

    cart_count = len(session.get('cart', []))
    search_history = session.get('search_history', [])

    return render_template(
        'search_results.html',
        results=paginated_results,
        query=query_str,
        cart_count=cart_count,
        search_history=search_history,
        current_page=page,
        total_pages=total_pages,
        start_page=max(1, page - 2),
        end_page=min(total_pages, page + 2)
    )

@app.route('/search-by-image', methods=['POST'])
def search_by_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = f'static/uploads/{uploaded_file.filename}'
        uploaded_file.save(file_path)
        similar_ids = image_search_engine.upload_and_search(file_path)
        uploaded_image_filename = uploaded_file.filename

        # Log first product_id to CSV and database
        if 'user_id' in session and similar_ids:
            user_id = session.get('user_id')
            log_to_csv(user_id, similar_ids[0], 'search')
            log_user_activity(user_id, similar_ids[0], 'search', quantity=1)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    similar_products = Product.query.filter(Product.product_id.in_(similar_ids)).all()
    tracking_data = Tracking.query.filter(Tracking.product_id.in_(similar_ids)).all()

    cart_count = len(session.get('cart', []))
    search_history = session.get('search_history', [])

    return render_template(
        'search_image.html',
        similar_products=zip(similar_products, tracking_data),
        uploaded_image_filename=uploaded_image_filename,
        cart_count=cart_count,
        search_history=search_history
    )

# Recommendation Routes
@app.route('/show_recommended/<int:page>')
def show_recommended(page=1):
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login_user'))

    per_page = 48
    start = (page - 1) * per_page

    recommended_products_with_tracking = get_recommendations(user_id)

    # Pagination
    total_products = len(recommended_products_with_tracking)
    total_pages = (total_products + per_page - 1) // per_page
    paginated_products = recommended_products_with_tracking[start:start + per_page]

    # Favourite status
    favourite_status = {
        product.Product.product_id: is_favourited(user_id, product.Product.product_id)
        for product in paginated_products
    }

    cart_count = len(session.get('cart', []))
    search_history = session.get('search_history', [])

    return render_template(
        'recommended.html',
        products=paginated_products,
        favourite_status=favourite_status,
        cart_count=cart_count,
        search_history=search_history,
        current_page=page,
        total_pages=total_pages,
        start_page=max(1, page - 2),
        end_page=min(total_pages, page + 2)
    )

@app.route('/content-based/<int:product_id>', methods=['GET'])
@app.route('/content-based/<int:product_id>/<int:page>')
def content_based_recommendations(product_id, page=1):
    per_page = 24

    selected_product = Product.query.get_or_404(product_id)
    product_category = ProductCategory.query.filter_by(product_id=product_id).first()

    if not product_category:
        return jsonify({'message': 'Product category not found.'}), 404

    category_type = categorize_product(product_category.category_id)
    similar_ids = content_based_filter.recommend(
        product_id=product_id,
        product_type=category_type,
        top_k=10,
        exclude_viewed=True,
        viewed_product_ids=None,
        content_based=True
    )

    similar_products = Product.query.filter(
        Product.product_id.in_(similar_ids)
    ).all()
    tracking_data = Tracking.query.filter(
        Tracking.product_id.in_([p.product_id for p in similar_products])
    ).all()

    total_products = len(similar_products)
    total_pages = (total_products + per_page - 1) // per_page
    paginated_products = similar_products[(page - 1) * per_page: page * per_page]

    cart_count = len(session.get('cart', []))
    search_history = session.get('search_history', [])

    return render_template(
        'content_based.html',
        cart_count=cart_count,
        selected_product=selected_product,
        similar_products=zip(paginated_products, tracking_data),
        current_page=page,
        total_pages=total_pages,
        start_page=max(1, page - 2),
        end_page=min(total_pages, page + 2),
        search_history=search_history,
        page_route='content_based'
    )

# Product Interaction Routes
@app.route('/favourite', methods=['POST'])
def favourite_route():
    data = request.get_json()
    product_id = data.get('product_id')
    activity_type = data.get('activity_type')

    # Get user_id from the session
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({'error': 'User not logged in'}), 401

    if activity_type == 'favourite':
        # Log 'favourite' activity
        log_user_activity(user_id, product_id, 'favourite')
        log_to_csv(user_id, product_id, 'favourite')
        return jsonify({'success': True, 'message': 'Product favourited'})
    elif activity_type == 'unfavourite':
        # Remove 'favourite' activity
        activity = UserActivityLog.query.filter_by(
            user_id=user_id, product_id=product_id, activity_type='favourite'
        ).first()
        if activity:
            db.session.delete(activity)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Product unfavourited'})
        else:
            return jsonify({'error': 'Favourite activity not found'}), 404
    else:
        return jsonify({'error': 'Invalid activity type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
