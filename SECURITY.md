# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public GitHub issue**
2. Open a [private security advisory](https://github.com/warlockee/Catapult/security/advisories/new) on GitHub
3. Include: description, steps to reproduce, potential impact

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Security Best Practices

- Change default `API_KEY_SALT` and `POSTGRES_PASSWORD` in `.env`
- Use HTTPS in production (configure SSL in `infrastructure/nginx/ssl/`)
- Restrict Docker socket access to trusted users
- Review API key permissions (admin vs viewer roles)
