# Standard library imports
import os
import csv
import json
import threading
import time
from datetime import datetime

# Third-party imports
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    jsonify, flash, current_app
)

# Local application imports
from db.dbo import (
    Product, Tracking, User, Category, ProductCategory,
    UserActivityLog, ActivityTypeEnum, db
)
from search_engine import ImageSearch, TextSearch
from recommendation_system import DeepContentBasedFiltering, DeepQNetwork

# Create Flask application
app = Flask(__name__)

# Configuration and constants
app.secret_key = 'Cam_on_Thay_Hien_Phan'  # Secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user_name:password2@localhost:5432/dbo'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db.init_app(app)

# Load configuration from JSON file
def load_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        return json.load(config_file)

config = load_config('model.json')

# Constants
MODEL_PATH = config['MODEL_PATH']
CSV_FILE_PATH = config['CSV_FILE_PATH']
MODEL_UPDATE_INTERVAL = 120  # Time interval to update DQN model (delta t) in seconds

# Ensure CSV file exists and has headers
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'product_id', 'activity_type', 'timestamp'])

# Initialize search and recommendation engines
image_search_engine = ImageSearch(config_file_path='model.json')
text_search_engine = TextSearch()
content_based_filter = DeepContentBasedFiltering('model.json')

# Initialize DQN agent
state_size = 10  # State size, can be adjusted
action_size = 50  # Assume 50 actions (e.g., popular products)
dqn_agent = DeepQNetwork(state_size=state_size, action_size=action_size)

# Load DQN model if it exists
dqn_agent.load(MODEL_PATH)

# Helper Functions

def calculate_reward(log):
    if log.activity_type == ActivityTypeEnum.favourite:
        return 2
    elif log.activity_type == ActivityTypeEnum.select:
        return 3
    elif log.activity_type == ActivityTypeEnum.remove_from_cart:
        return -1
    else:
        return 1  # Default reward for view action

def calculate_reward_log_type(activity_type):
    score_mapping = {
        'favourite': 2,
        'select': 3,
        'remove_from_cart': -1,
        'search': 1,
        'view': 1
    }
    return score_mapping.get(activity_type, 0)

def extract_state_from_db(user_id):
    logs = UserActivityLog.query.filter_by(user_id=user_id).order_by(
        UserActivityLog.view_end_time.desc()
    ).limit(state_size).all()
    state = [log.product_id for log in logs]
    state += [0] * (state_size - len(state))  # Padding if needed
    return np.array(state)

def log_to_csv(user_id, product_id, activity_type):
    with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, product_id, activity_type, datetime.now()])

def log_user_activity(user_id, product_id, activity_type, quantity=1, view_end_time=None):
    valid_activity_types = ['view', 'select', 'purchase', 'remove_from_cart', 'favourite']
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

def favourited(user_id, product_id):
    if user_id:
        favourite_entry = UserActivityLog.query.filter_by(
            user_id=user_id, product_id=product_id, activity_type='favourite'
        ).first()
        return bool(favourite_entry)
    return False

def get_activity_score(activity_type):
    score_mapping = {
        ActivityTypeEnum.view: 1,
        ActivityTypeEnum.favourite: 3,
        ActivityTypeEnum.select: 2,
        ActivityTypeEnum.remove_from_cart: -1
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

# Background Tasks

def update_dqn_model():
    with app.app_context():
        while True:
            print("Bắt đầu huấn luyện DQN sau mỗi khoảng thời gian delta t...")

            # Get list of user_ids from database
            users = User.query.all()

            for user in users:
                user_id = user.user_id

                # Extract state and train model for each user
                state = extract_state_from_db(user_id).reshape(1, -1)
                action = dqn_agent.act(state)
                reward = calculate_reward_log_type('some_activity')
                next_state = extract_state_from_db(user_id).reshape(1, -1)

                done = False  # Define end condition if necessary

                # Train DQN model
                dqn_agent.remember(state, action, reward, next_state, done)
                dqn_agent.replay()

            # Save model after training
            dqn_agent.save(MODEL_PATH)
            print(f"Mô hình DQN đã được cập nhật và lưu tại {MODEL_PATH}")

            # Wait for delta t before next training
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

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))

    user_id = session.get('user_id')

    # Get user's interaction history
    activity_logs = UserActivityLog.query.filter_by(
        user_id=user_id
    ).order_by(UserActivityLog.view_end_time.desc()).limit(5).all()

    favorite_products = [
        log.product_id for log in activity_logs if log.activity_type == ActivityTypeEnum.favourite
    ]
    viewed_products = [
        log.product_id for log in activity_logs
        if log.activity_type == ActivityTypeEnum.view and log.product_id not in favorite_products
    ]

    # Combine product IDs for recommendation
    product_ids_for_recommendation = favorite_products + viewed_products[:max(0, 5 - len(favorite_products))]

    recommended_product_ids = []

    # Get recommendations from DQN based on state
    state = extract_state_from_db(user_id).reshape(1, -1)
    action = dqn_agent.act(state)

    # Assume action is index in product_ids_for_recommendation
    if product_ids_for_recommendation:
        selected_product_id = product_ids_for_recommendation[action % len(product_ids_for_recommendation)]
        # Get similar products using content-based filter
        product_category = ProductCategory.query.filter_by(product_id=selected_product_id).first()
        if product_category:
            category_type = categorize_product(product_category.category_id)
            similar_ids = content_based_filter.recommend(
                product_id=selected_product_id,
                product_type=category_type,
                top_k=3,
                exclude_viewed=True,
                viewed_product_ids=[
                    log.product_id for log in activity_logs if log.activity_type == ActivityTypeEnum.view
                ]
            )
            recommended_product_ids.extend(similar_ids)

    # Recommend 3 similar products for viewed products
    for product_id in viewed_products[:3]:
        product_category = ProductCategory.query.filter_by(product_id=product_id).first()
        if product_category:
            category_type = categorize_product(product_category.category_id)
            similar_ids = content_based_filter.recommend(
                product_id=product_id,
                product_type=category_type,
                top_k=3,
                exclude_viewed=True,
                viewed_product_ids=[
                    log.product_id for log in activity_logs if log.activity_type == ActivityTypeEnum.view
                ]
            )
            recommended_product_ids.extend(similar_ids)

    # Recommend 3 similar products for favourite products
    for product_id in favorite_products[:3]:
        product_category = ProductCategory.query.filter_by(product_id=product_id).first()
        if product_category:
            category_type = categorize_product(product_category.category_id)
            similar_ids = content_based_filter.recommend(
                product_id=product_id,
                product_type=category_type,
                top_k=3,
                exclude_viewed=True,
                viewed_product_ids=[
                    log.product_id for log in activity_logs if log.activity_type == ActivityTypeEnum.view
                ]
            )
            recommended_product_ids.extend(similar_ids)

    # Remove duplicates and limit recommendations
    recommended_product_ids = list(dict.fromkeys(recommended_product_ids))[:9]

    # Get recommended products from database
    recommended_products = (
        db.session.query(Product, Tracking)
        .join(Tracking, Tracking.product_id == Product.product_id)
        .filter(Product.product_id.in_(recommended_product_ids))
        .all()
    )

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
        product.Product.product_id: favourited(user_id, product.Product.product_id)
        for product in recommended_products
    }
    favourite_status_top_rated = {
        product.Product.product_id: favourited(user_id, product.Product.product_id)
        for product in top_rated_products
    }

    cart_count = len(session.get('cart', []))
    search_history = session.get('search_history', [])

    return render_template(
        'home.html',
        recommended_products=recommended_products,
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
        product.Product.product_id: favourited(user_id, product.Product.product_id)
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
    favourite_type = {product.product_id: favourited(user_id, product.product_id)}

    product_category = ProductCategory.query.filter_by(product_id=product_id).first()
    category_type = categorize_product(product_category.category_id)
    similar_ids = content_based_filter.recommend(
        product_id=product_id,
        product_type=category_type,
        top_k=9,
        exclude_viewed=True,
        viewed_product_ids=None
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
    favourite_status_top_rated = {
        product.product_id: favourited(user_id, product.product_id)
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
        favourite_type=favourite_type,
        tracking=tracking,
        highlight=highlight,
        similar_products=products_with_tracking,
        favourite_status=favourite_status_top_rated
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

    # Ensure price is properly formatted and converted to float
    def clean_price(price):
        try:
            return float(str(price).replace(',', '').replace('VND', '').strip())
        except ValueError:
            return 0.0

    total = 0.0
    for item in grouped_cart_items.values():
        price = clean_price(item['price'])
        quantity = int(item.get('quantity', 1))
        total += price * quantity

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
        product_id = request.form.get('product_id')
        quantity = int(request.form.get('quantity', 1))
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
        # Get search history from the session
        search_history = session.setdefault('search_history', [])

        if query_str not in search_history:
            search_history.insert(0, query_str)

        # Keep only the 5 most recent unique search terms
        session['search_history'] = search_history[:5]

        # Proceed with search logic
        results = text_search_engine.search(query_str)

        formatted_results = [
            (
                Product.query.filter_by(product_id=product_id).first(),
                Tracking.query.filter_by(product_id=product_id).first()
            )
            for product_id, _, _, _, _ in results
            if Product.query.filter_by(product_id=product_id).first() and
            Tracking.query.filter_by(product_id=product_id).first()
        ]

        # Log first 3 product_ids to CSV when searching
        if 'user_id' in session:
            user_id = session.get('user_id')
            for product, _ in formatted_results[:3]:
                log_to_csv(user_id, product.product_id, 'search')

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

        # Log first product_id to CSV when searching by image
        if 'user_id' in session and similar_ids:
            user_id = session.get('user_id')
            log_to_csv(user_id, similar_ids[0], 'search')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    similar_products = Product.query.filter(
        Product.product_id.in_(similar_ids)
    ).all()
    tracking_data = Tracking.query.filter(
        Tracking.product_id.in_(similar_ids)
    ).all()

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

import os

@app.route('/recommended/<int:page>')
def show_recommended(page=1):
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login_user'))

    per_page = 48
    start = (page - 1) * per_page

    activity_logs = []

    # Check if the CSV file exists and is not empty
    if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
        with open(CSV_FILE_PATH, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            try:
                next(reader)  # Skip header
            except StopIteration:
                pass  # CSV file is empty after header
            else:
                for row in reader:
                    if row[0] == str(user_id):
                        activity_logs.append((int(row[1]), row[2]))  # (product_id, activity_type)
    else:
        # CSV file does not exist or is empty
        pass  # activity_logs remains empty

    # Prioritize group 1: select, favourite, view, search
    select_products = [log[0] for log in activity_logs if log[1] == 'select']
    favourite_products = [log[0] for log in activity_logs if log[1] == 'favourite']
    view_products = [log[0] for log in activity_logs if log[1] == 'view']
    search_products = [log[0] for log in activity_logs if log[1] == 'search']

    # Combine product_ids for group 1 in order of priority
    interacted_product_ids = select_products + favourite_products + view_products + search_products

    # Get recommendations from DQN (Group 2)
    dqn_recommended_ids = []
    state = extract_state_from_db(user_id).reshape(1, -1)
    action = dqn_agent.act(state)

    if interacted_product_ids:
        selected_product_id = interacted_product_ids[action % len(interacted_product_ids)]
        dqn_recommended_ids.append(selected_product_id)

    # Get similar products to those in group 1 (Group 3)
    similar_product_ids = []
    for product_id in interacted_product_ids:
        product_category = ProductCategory.query.filter_by(product_id=product_id).first()
        if product_category:
            category_type = categorize_product(product_category.category_id)
            similar_ids = content_based_filter.recommend(
                product_id=product_id,
                product_type=category_type,
                top_k=15,  # Adjust as needed
                exclude_viewed=False
            )
            similar_product_ids.extend(similar_ids)

    # Remove duplicates between groups
    dqn_recommended_ids = list(dict.fromkeys(dqn_recommended_ids))
    similar_product_ids = list(dict.fromkeys(similar_product_ids))

    # Combine all product_ids
    all_recommended_ids = list(dict.fromkeys(
        interacted_product_ids + dqn_recommended_ids + similar_product_ids
    ))

    # Get products from database
    recommended_products_with_tracking = (
        db.session.query(Product, Tracking)
        .join(Tracking, Tracking.product_id == Product.product_id)
        .filter(Product.product_id.in_(all_recommended_ids))
        .all()
    )

    # Pagination
    total_products = len(recommended_products_with_tracking)
    total_pages = (total_products + per_page - 1) // per_page
    paginated_products = recommended_products_with_tracking[start:start + per_page]

    favourite_status = {
        product.Product.product_id: favourited(user_id, product.Product.product_id)
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
        viewed_product_ids=None
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