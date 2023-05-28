import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import cloudinary
import cloudinary.uploader
from app_gradio import process
from bs4 import BeautifulSoup

cloudinary.config(
    cloud_name="dtvgddjmz",
    api_key="181189433131294",
    api_secret="AbcY3VYoe7YyY2gJExum3_J1HoI"
)

# Model for Post
class Post(BaseModel):
    title: str = Field(..., description="Title of the post", min_length=1)
    content: str = Field(..., description="Content of the post", min_length=1)
    date: str
    category: str = Field(..., description="Category of the post", min_length=1)
    imageUrl: Optional[str] = None
    audioUrl: Optional[str] = None
    views: int = 1

# FastAPI app
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

day_mapping = {
    "Monday": "Thứ Hai",
    "Tuesday": "Thứ Ba",
    "Wednesday": "Thứ Tư",
    "Thursday": "Thứ Năm",
    "Friday": "Thứ Sáu",
    "Saturday": "Thứ Bảy",
    "Sunday": "Chủ Nhật"
}

# MongoDB service
class MongoDBService:
    def __init__(self):
        self.client = MongoClient("mongodb+srv://hoajtranh:65Xez8WGBMARUmFA@cluster0.uhxhrki.mongodb.net/")
        self.db = self.client.get_database("TTS")
        self.posts_collection = self.db.get_collection("posts")

    def get_collection(self):
        return self.posts_collection

# Dependency for MongoDBService
def get_mongodb_service():
    service = MongoDBService()
    try:
        yield service
    finally:
        service.client.close()

# Create a post
@app.post("/posts")
def create_post(post: Post, mongodb_service: MongoDBService = Depends(get_mongodb_service)):
    # Kiểm tra xem đã có bài viết với title đã cho hay chưa
    existing_post = mongodb_service.get_collection().find_one({"title": post.title})
    if existing_post:
        raise HTTPException(status_code=400, detail="A post with the same title already exists")
        
    post_dict = post.dict()
    post_dict["date"] = datetime.now().strftime("%A %d/%m/%Y - %H:%M")

    # Xử lý văn bản và tạo file âm thanh
    title_text = post_dict["title"]
    content_text = post_dict["content"]

    soup = BeautifulSoup(content_text, "html.parser")
    content_plain_text = soup.get_text()

    combined_text = f"{title_text}. {content_plain_text}"

    audio_path = process(combined_text)

    # Upload file âm thanh lên Cloudinary
    audio_response = cloudinary.uploader.upload(audio_path, resource_type="auto")

    # Lấy đường dẫn an toàn từ response và gán cho thuộc tính audioUrl
    post_dict["audioUrl"] = audio_response["secure_url"]

    # Xóa file âm thanh tạm thời
    os.remove(audio_path)

    result = mongodb_service.get_collection().insert_one(post_dict)
    post_dict["_id"] = str(result.inserted_id)
    return post_dict

# Get all posts
@app.get("/posts")
def get_all_posts(mongodb_service: MongoDBService = Depends(get_mongodb_service)):
    posts = mongodb_service.get_collection().find()
    formatted_posts = []
    for post in posts:
        date_parts = post["date"].split(" ")
        day_of_week = day_mapping.get(date_parts[0], date_parts[0])
        formatted_date = f"{day_of_week} {date_parts[1]} - {date_parts[3]}"
        post["date"] = formatted_date
        formatted_posts.append(Post(**post))
    return formatted_posts

# Get post by title
@app.get("/posts/{title}")
def get_post_by_title(title: str, mongodb_service: MongoDBService = Depends(get_mongodb_service)):
    post = mongodb_service.get_collection().find_one({"title": title})
    if post:
        # Tăng giá trị của trường "views"
        mongodb_service.get_collection().update_one({"_id": post["_id"]}, {"$inc": {"views": 0.5}})
        
        date_parts = post["date"].split(" ")
        day_of_week = day_mapping.get(date_parts[0], date_parts[0])
        formatted_date = f"{day_of_week} {date_parts[1]} - {date_parts[3]}"
        post["date"] = formatted_date
        return Post(**post)
    else:
        raise HTTPException(status_code=404, detail="Post not found")

# Get posts by category
@app.get("/posts/category/{category}")
def get_posts_by_category(category: str, mongodb_service: MongoDBService = Depends(get_mongodb_service)):
    posts = mongodb_service.get_collection().find({"category": category})
    formatted_posts = []
    for post in posts:
        date_parts = post["date"].split(" ")
        day_of_week = day_mapping.get(date_parts[0], date_parts[0])
        formatted_date = f"{day_of_week} {date_parts[1]} - {date_parts[3]}"
        post["date"] = formatted_date
        formatted_posts.append(Post(**post))
    return formatted_posts

# Update post by ID
@app.put("/posts/{post_id}")
def update_post(post_id: str, post: Post, mongodb_service: MongoDBService = Depends(get_mongodb_service)):
    updated_post = post.dict(exclude_unset=True)
    result = mongodb_service.get_collection().update_one({"_id": ObjectId(post_id)}, {"$set": updated_post})
    if result.modified_count == 1:
        return {"message": "Post updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Post not found")

# Delete post by ID
@app.delete("/posts/{post_id}")
def delete_post(post_id: str, mongodb_service: MongoDBService = Depends(get_mongodb_service)):
    result = mongodb_service.get_collection().delete_one({"_id": ObjectId(post_id)})
    if result.deleted_count == 1:
        return {"message": "Post deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Post not found")
