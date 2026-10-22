"""Persistent local accounts and revocable opaque cookie sessions.

Account mode isolates previous local records instead of assigning them to the
first registrant. The HTTP process never persists or returns plaintext secrets.
"""
import hashlib
import hmac
import ipaddress
import secrets
import sqlite3
import time

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from src.api.jobs import LOCAL_OWNER, utc_now

COOKIE = 'tennis_session'
SESSION_SECONDS = 7 * 86400


class Credentials(BaseModel):
    model_config = ConfigDict(extra='forbid')
    username: str = Field(min_length=3, max_length=80, pattern=r'^[A-Za-z0-9_.-]+$')
    password: str = Field(min_length=12, max_length=256)


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


class AuthStore:
    def __init__(self, jobs, settings, public_read=None):
        self.jobs, self.settings = jobs, settings
        self.public_read = public_read or (lambda request: False)
        with jobs.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                password_salt TEXT NOT NULL, password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                expires_at INTEGER NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS auth_attempts (
                client TEXT PRIMARY KEY, count INTEGER NOT NULL, window_start INTEGER NOT NULL)''')

    def user(self, request):
        token = request.cookies.get(COOKIE)
        if not token or len(token) > 200:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.jobs.connect() as conn:
            row = conn.execute('''SELECT a.id,a.username FROM auth_sessions s
                JOIN accounts a ON a.id=s.owner_id WHERE s.token_hash=? AND s.expires_at>?''',
                               (digest, int(time.time()))).fetchone()
        return dict(row) if row else None

    def session(self, response, user):
        token = secrets.token_urlsafe(32)
        with self.jobs.connect() as conn:
            conn.execute('DELETE FROM auth_sessions WHERE expires_at<=?', (int(time.time()),))
            conn.execute('INSERT INTO auth_sessions VALUES (?,?,?)',
                         (hashlib.sha256(token.encode()).hexdigest(), user['id'], int(time.time()) + SESSION_SECONDS))
        response.set_cookie(COOKIE, token, httponly=True, samesite='strict',
                            secure=self.settings.secure_cookies, max_age=SESSION_SECONDS, path='/')

    def throttle(self, request):
        client = request.client.host if request.client else 'unknown'
        now = int(time.time())
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT * FROM auth_attempts WHERE client=?', (client,)).fetchone()
            if row and row['window_start'] > now - 300 and row['count'] >= 20:
                raise HTTPException(429, 'Too many sign-in attempts. Try again in a few minutes.')
            count = row['count'] + 1 if row and row['window_start'] > now - 300 else 1
            start = row['window_start'] if row and row['window_start'] > now - 300 else now
            conn.execute('INSERT OR REPLACE INTO auth_attempts VALUES (?,?,?)', (client, count, start))

    def install(self, app):
        app.add_middleware(AccessMiddleware, auth=self)

        router = APIRouter(prefix='/api/auth')

        @router.get('/session')
        def session(request: Request):
            user = self.user(request)
            return {'required': self.settings.require_auth, 'authenticated': bool(user), 'user': user}

        @router.post('/register', status_code=201)
        def register(payload: Credentials, request: Request, response: Response):
            self.throttle(request)
            salt = secrets.token_hex(16)
            user = {'id': 'account_' + secrets.token_hex(16), 'username': payload.username.lower()}
            digest = password_hash(payload.password, salt)
            try:
                with self.jobs.connect() as conn:
                    conn.execute('INSERT INTO accounts VALUES (?,?,?,?,?)',
                                 (user['id'], user['username'], salt, digest, utc_now()))
            except sqlite3.IntegrityError as exc:
                raise HTTPException(409, 'This username is unavailable.') from exc
            self.session(response, user)
            return {'required': self.settings.require_auth, 'authenticated': True, 'user': user}

        @router.post('/login')
        def login(payload: Credentials, request: Request, response: Response):
            self.throttle(request)
            with self.jobs.connect() as conn:
                row = conn.execute('SELECT * FROM accounts WHERE username=?', (payload.username.lower(),)).fetchone()
            salt = row['password_salt'] if row else '0' * 32
            digest = password_hash(payload.password, salt)
            if not row or not hmac.compare_digest(digest, row['password_hash']):
                raise HTTPException(401, 'Username or password is incorrect.')
            user = {'id': row['id'], 'username': row['username']}
            self.session(response, user)
            return {'required': self.settings.require_auth, 'authenticated': True, 'user': user}

        @router.post('/logout')
        def logout(request: Request, response: Response):
            token = request.cookies.get(COOKIE, '')
            with self.jobs.connect() as conn:
                conn.execute('DELETE FROM auth_sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))
            response.delete_cookie(COOKIE, path='/')
            return {'authenticated': False}

        app.include_router(router)


class AccessMiddleware:
    """Pure ASGI authorization preserves owned streaming/disconnect behavior."""
    def __init__(self, app, auth):
        self.app, self.auth = app, auth

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api/'):
            return await self.app(scope, receive, send)
        from starlette.datastructures import MutableHeaders
        from starlette.responses import JSONResponse
        from starlette.concurrency import run_in_threadpool

        request = Request(scope, receive)
        request.state.owner_id = LOCAL_OWNER
        user = await run_in_threadpool(self.auth.user, request)
        request.state.user = user
        host = request.client.host if request.client else ''
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == 'testclient'
        curated = await run_in_threadpool(self.auth.public_read, request)
        error, status = None, 403
        if self.auth.settings.require_auth:
            if not local and (not self.auth.settings.secure_cookies or request.url.scheme != 'https'):
                error = 'Remote account access requires TENNIS_SECURE_COOKIES=1 and HTTPS.'
            public = request.url.path in {'/api/health', '/api/auth/session', '/api/auth/login', '/api/auth/register'} or curated
            if not error and not public and not user:
                error, status = 'Sign in to access your matches.', 401
            if user:
                request.state.owner_id = user['id']
        else:
            if not local:
                error = 'Remote access requires TENNIS_REQUIRE_AUTH=1.'
            hostname = request.url.hostname or ''
            try:
                loopback_host = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback_host = hostname == 'localhost' or (host == 'testclient' and hostname == 'testserver')
            if not error and not loopback_host:
                error = 'Local mode requires a loopback hostname.'
        origin = request.headers.get('origin')
        if not error and request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin and origin != f'{request.url.scheme}://{request.url.netloc}':
            error = 'Cross-origin changes are not allowed.'
        if error:
            response = JSONResponse({'detail': error}, status_code=status, headers={'Cache-Control': 'private, no-store'})
            return await response(scope, receive, send)

        async def with_private_cache(message):
            if message['type'] == 'http.response.start':
                headers = MutableHeaders(scope=message)
                # A curated bare URL may select the owner's newer private
                # generation. Authentication must never relax private caching.
                if user or (self.auth.settings.require_auth and not curated) or 'Cache-Control' not in headers:
                    headers['Cache-Control'] = 'private, no-store'
                if self.auth.settings.require_auth and curated:
                    headers.add_vary_header('Cookie')
            await send(message)
        return await self.app(scope, receive, with_private_cache)


def current_owner(request: Request):
    return request.state.owner_id
