import aiosqlite
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from backend.config import get_settings
from backend.api.schemas.auth import UserRegister, UserLogin, Token, UserResponse
from backend.api.auth_utils import get_password_hash, verify_password, create_access_token, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Auth"])

@router.post("/register", response_model=Token)
async def register(user: UserRegister):
    settings = get_settings()
    hashed_password = get_password_hash(user.password)
    
    try:
        async with aiosqlite.connect(settings.sqlite_db_path) as db:
            # Check if exists
            async with db.execute("SELECT id FROM users WHERE email = ?", (user.email,)) as cursor:
                if await cursor.fetchone():
                    raise HTTPException(status_code=400, detail="Email already registered")
                    
            await db.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (user.name, user.email, hashed_password)
            )
            await db.commit()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
        
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    settings = get_settings()
    
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        async with db.execute("SELECT password_hash FROM users WHERE email = ?", (user.email,)) as cursor:
            row = await cursor.fetchone()
            if not row or not verify_password(user.password, row[0]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user
