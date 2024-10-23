from flask_sqlalchemy import SQLAlchemy
from enum import Enum
from sqlalchemy.sql import func

db = SQLAlchemy()

# User model
class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now(), nullable=False)
    updated_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

# Categories table
class Category(db.Model):
    __tablename__ = 'categories'

    category_id = db.Column(db.Integer, primary_key=True)
    category_name = db.Column(db.String(255), nullable=False, unique=True)
    parent_category_id = db.Column(db.Integer, db.ForeignKey('categories.category_id'), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    # Self-referencing relationship for parent categories
    parent_category = db.relationship('Category', remote_side=[category_id], backref='subcategories')

# Product table
class Product(db.Model):
    __tablename__ = 'products'
 
    product_id = db.Column(db.Integer, primary_key=True)
    sku = db.Column(db.String(100), unique=True, nullable=False)
    product_name = db.Column(db.Text, nullable=False)
    product_description = db.Column(db.Text)
    product_price = db.Column(db.Numeric(10, 2), nullable=False)
    product_images = db.Column(db.ARRAY(db.Text))  # Array of image URLs
    product_highlights = db.Column(db.ARRAY(db.Text))  # Array of highlights
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

    # Relationship with tracking (one-to-one)
    tracking = db.relationship('Tracking', backref='product', uselist=False)

    # Many-to-many relationship with authors
    authors = db.relationship('Author', secondary='product_authors', backref=db.backref('products', lazy=True))

    # Many-to-many relationship with categories
    categories = db.relationship('Category', secondary='product_categories', backref=db.backref('products', lazy=True))

# Tracking table
class Tracking(db.Model):
    __tablename__ = 'trackings'

    track_id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'), unique=True)
    discount = db.Column(db.Numeric(5, 2), default=0)
    quantity_sold = db.Column(db.Integer, default=0)
    review_count = db.Column(db.Integer, default=0)
    rating_average = db.Column(db.Numeric(3, 2), default=0)
    favorite_count = db.Column(db.Integer, default=0)

# Author table
class Author(db.Model):
    __tablename__ = 'authors'

    author_id = db.Column(db.Integer, primary_key=True)
    author_name = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

# ProductCategories association table
class ProductCategory(db.Model):
    __tablename__ = 'product_categories'

    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'), primary_key=True)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.category_id'), primary_key=True)

# ProductAuthors association table
class ProductAuthor(db.Model):
    __tablename__ = 'product_authors'

    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'), primary_key=True)
    author_id = db.Column(db.Integer, db.ForeignKey('authors.author_id'), primary_key=True)

# Attributes model
class Attribute(db.Model):
    __tablename__ = 'attributes'
    attribute_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id', ondelete="CASCADE"), nullable=False, unique=True)
    publisher = db.Column(db.Text)
    translator = db.Column(db.Text)
    cover_type = db.Column(db.Text)
    number_pages = db.Column(db.Integer, db.CheckConstraint('number_pages >= 0'), nullable=True)
    publishing_house = db.Column(db.Text)
    publication_date = db.Column(db.Date)
    dimensions = db.Column(db.Numeric(5, 2))  # Adjust this if you need array dimensions in a different way
    created_at = db.Column(db.TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = db.Column(db.TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


# Use Python's Enum for activity type
class ActivityTypeEnum(Enum):
    view = 'view'
    select = 'select'
    purchase = 'purchase'
    remove_from_cart = 'remove_from_cart'
    favourite = 'favourite'

# SQLAlchemy Enum should reference the Python Enum
activity_type_enum = db.Enum(ActivityTypeEnum)

# Mô hình bảng user_activity_logs
class UserActivityLog(db.Model):
    __tablename__ = 'user_activity_logs'

    activity_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'), nullable=False)
    activity_type = db.Column(activity_type_enum, nullable=False)
    view_start_time = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    view_end_time = db.Column(db.DateTime(timezone=True), nullable=True)
    quantity = db.Column(db.Integer, default=1)