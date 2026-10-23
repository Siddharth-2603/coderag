# Authentication

The service uses token-based authentication. Clients obtain a JWT by calling
the `/login` endpoint with a username and password. The token is signed with a
server-side secret and expires after 24 hours.

Every protected endpoint checks the `Authorization: Bearer <token>` header.
If the token is missing, expired, or has an invalid signature, the request is
rejected with a 401 Unauthorized response.

Refresh tokens are issued alongside access tokens and can be exchanged at
`/refresh` to obtain a new access token without re-entering credentials.
