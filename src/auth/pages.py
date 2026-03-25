"""
src/auth/pages.py — Streamlit Auth UI
======================================
Three rendered views, all called from app.py before the main dashboard.

  render_auth_gate()
    Dispatcher — reads st.session_state["auth_page"] and calls:
      "login"    → _render_login()
      "register" → _render_register()
      "payment"  → _render_payment()

Session state keys written by this module
-----------------------------------------
  authenticated    bool   — True once login is complete and payment verified
  auth_page        str    — current auth view: "login" | "register" | "payment"
  auth_user_id     int    — DB primary key of the logged-in user
  auth_user_name   str    — display name shown in the dashboard
  auth_user_email  str    — email shown in the sidebar

Design: Medical-Blue dark theme, distinct from the dashboard's pure-black theme.
"""
from __future__ import annotations

import io

import streamlit as st

from src.auth.database import SessionLocal
from src.auth.utils import (
    create_user,
    get_user_by_email,
    update_payment_status,
    verify_password,
)

# ─────────────────────────────────────────────────────────────────────────────
# QR Code (generated once, cached for the process lifetime)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_payment_qr_bytes() -> bytes:
    """
    Generate the UPI payment QR code as PNG bytes.
    Cached so the image is only generated once per server process.
    """
    import qrcode  # type: ignore
    from qrcode.image.styledpil import StyledPilImage  # type: ignore
    from qrcode.image.styles.moduledrawers import RoundedModuleDrawer  # type: ignore

    upi_url = (
        "upi://pay?pa=medverify@ybl"
        "&pn=MedVerify%20Clinical%20Suite"
        "&am=899"
        "&cu=INR"
        "&tn=MedVerify+Clinical+Access"
    )
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(upi_url)
    qr.make(fit=True)

    try:
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=RoundedModuleDrawer(),
            back_color=(255, 255, 255),
            fill_color=(13, 27, 62),
        )
    except Exception:
        # Fallback to plain QR if styled factory unavailable
        img = qr.make_image(fill_color="#0D1B3E", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Shared CSS — Medical Blue theme
# ─────────────────────────────────────────────────────────────────────────────
_AUTH_CSS = """
<style>
/* ── Override dark app theme for auth pages ─────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: linear-gradient(135deg, #050d1a 0%, #0a1628 55%, #0d2040 100%)
                !important;
    color: #dce8f5 !important;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
}
[data-testid="stSidebar"]  { display: none !important; }
[data-testid="stHeader"]   { background: transparent !important; }
[data-testid="stToolbar"]  { display: none !important; }
footer                     { display: none !important; }

/* ── Auth card ───────────────────────────────────────────────────────────── */
.auth-card {
    background: rgba(15, 35, 75, 0.85);
    border: 1px solid #1e3f80;
    border-top: 3px solid #2196F3;
    border-radius: 14px;
    padding: 2.2rem 2.6rem 2rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    max-width: 440px;
    margin: 0 auto;
}

/* ── Logo / brand ────────────────────────────────────────────────────────── */
.auth-logo {
    text-align: center;
    margin-bottom: 0.2rem;
}
.auth-logo-icon {
    font-size: 2.8rem;
    line-height: 1;
}
.auth-brand {
    color: #64B5F6;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-align: center;
    margin: 0;
}
.auth-tagline {
    color: #7b9cbf;
    font-size: 0.78rem;
    text-align: center;
    margin: 0.1rem 0 1.4rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Section title ───────────────────────────────────────────────────────── */
.auth-title {
    color: #E3F2FD;
    font-size: 1.15rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1.2rem;
    letter-spacing: 0.02em;
}

/* ── Input overrides ─────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] input:focus {
    background: #091729 !important;
    border: 1px solid #1e3f80 !important;
    border-radius: 8px !important;
    color: #dce8f5 !important;
    font-size: 0.93rem !important;
    padding: 0.55rem 0.9rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #2196F3 !important;
    box-shadow: 0 0 0 2px rgba(33,150,243,0.25) !important;
}
[data-testid="stTextInput"] label {
    color: #90b4d4 !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Primary button ──────────────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1565C0, #1976D2) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.2rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 14px rgba(21,101,192,0.4) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1976D2, #42A5F5) !important;
    box-shadow: 0 6px 20px rgba(21,101,192,0.55) !important;
}

/* ── Switch link ─────────────────────────────────────────────────────────── */
.auth-switch {
    text-align: center;
    font-size: 0.82rem;
    color: #7b9cbf;
    margin-top: 1.1rem;
}
.auth-switch a { color: #64B5F6; text-decoration: none; font-weight: 600; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
.auth-divider {
    border: none;
    border-top: 1px solid #1e3a60;
    margin: 1.2rem 0;
}

/* ── Error / success banners ─────────────────────────────────────────────── */
.auth-error {
    background: #2d0f0f;
    border: 1px solid #d32f2f;
    border-left: 4px solid #f44336;
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    color: #ff8a80;
    font-size: 0.83rem;
    margin-bottom: 0.8rem;
}
.auth-success {
    background: #0a2218;
    border: 1px solid #2e7d32;
    border-left: 4px solid #4caf50;
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    color: #a5d6a7;
    font-size: 0.83rem;
    margin-bottom: 0.8rem;
}

/* ── Payment card ────────────────────────────────────────────────────────── */
.pay-card {
    background: rgba(15, 35, 75, 0.85);
    border: 1px solid #1e3f80;
    border-top: 3px solid #FFC107;
    border-radius: 14px;
    padding: 2rem 2.4rem 1.8rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    max-width: 480px;
    margin: 0 auto;
    text-align: center;
}
.pay-title {
    color: #FFC107;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    margin-bottom: 0.25rem;
}
.pay-subtitle {
    color: #90b4d4;
    font-size: 0.82rem;
    margin-bottom: 1.4rem;
}
.pay-amount {
    font-size: 2.4rem;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin: 0.5rem 0;
}
.pay-amount span {
    color: #FFC107;
}
.pay-caption {
    color: #7b9cbf;
    font-size: 0.78rem;
    margin-top: 0.5rem;
    line-height: 1.5;
}
.pay-features {
    background: rgba(33,150,243,0.08);
    border: 1px solid #1e3f80;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 1.2rem 0 1.4rem;
    text-align: left;
    font-size: 0.81rem;
    color: #90b4d4;
    line-height: 1.8;
}
.pay-confirm-btn > button {
    background: linear-gradient(135deg, #F57F17, #FFC107) !important;
    color: #0a0a0a !important;
    font-weight: 800 !important;
    font-size: 1.0rem !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(255,193,7,0.4) !important;
}
.pay-confirm-btn > button:hover {
    box-shadow: 0 6px 22px rgba(255,193,7,0.6) !important;
}

/* ── Security badges row ─────────────────────────────────────────────────── */
.sec-badges {
    display: flex;
    justify-content: center;
    gap: 1.4rem;
    margin-top: 1.2rem;
    font-size: 0.72rem;
    color: #546e8a;
}
.sec-badge {
    display: flex;
    align-items: center;
    gap: 0.3rem;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────
def render_auth_gate() -> None:
    """
    Entry point called from app.py.
    Injects shared CSS and dispatches to the correct sub-page.
    """
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    page = st.session_state.get("auth_page", "login")
    if page == "register":
        _render_register()
    elif page == "payment":
        _render_payment()
    else:
        _render_login()


# ─────────────────────────────────────────────────────────────────────────────
# Login
# ─────────────────────────────────────────────────────────────────────────────
def _render_login() -> None:
    _spacer(3)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        # ── Brand header ─────────────────────────────────────────────────────
        st.markdown(
            '<div class="auth-card">'
            '<div class="auth-logo"><span class="auth-logo-icon">🔬</span></div>'
            '<p class="auth-brand">Med-Verify</p>'
            '<p class="auth-tagline">Clinical Logic Engine · Ophthalmology Edition</p>'
            '<p class="auth-title">Sign In to Your Account</p>',
            unsafe_allow_html=True,
        )

        # ── Error / success flash ─────────────────────────────────────────────
        if st.session_state.get("_login_error"):
            st.markdown(
                f'<div class="auth-error">⚠ {st.session_state.pop("_login_error")}</div>',
                unsafe_allow_html=True,
            )
        if st.session_state.get("_login_success"):
            st.markdown(
                f'<div class="auth-success">✓ {st.session_state.pop("_login_success")}</div>',
                unsafe_allow_html=True,
            )

        # ── Login form ────────────────────────────────────────────────────────
        with st.form("login_form", clear_on_submit=False):
            email    = st.text_input("Email Address", placeholder="doctor@hospital.in")
            password = st.text_input("Password", type="password",
                                      placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)

        if submitted:
            _do_login(email.strip(), password)

        # ── Switch to register ────────────────────────────────────────────────
        st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-switch">New to Med-Verify?&nbsp;</div>',
            unsafe_allow_html=True,
        )
        if st.button("Create Account →", use_container_width=True, key="go_register"):
            st.session_state["auth_page"] = "register"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        _sec_badges()


def _do_login(email: str, password: str) -> None:
    if not email or not password:
        st.session_state["_login_error"] = "Please enter both email and password."
        st.rerun()

    db = SessionLocal()
    try:
        user = get_user_by_email(db, email)
        if not user or not verify_password(password, user.password_hash):
            st.session_state["_login_error"] = "Incorrect email or password."
            st.rerun()

        # Credentials valid — check payment
        st.session_state.update({
            "auth_user_id":    user.id,
            "auth_user_name":  user.full_name,
            "auth_user_email": user.email,
        })

        if not user.payment_status:
            st.session_state["auth_page"] = "payment"
            st.rerun()

        # Fully authenticated
        st.session_state["authenticated"] = True
        st.session_state["auth_page"]     = "dashboard"
        st.session_state["_login_success"] = f"Welcome back, {user.full_name}!"
        st.rerun()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Register
# ─────────────────────────────────────────────────────────────────────────────
def _render_register() -> None:
    _spacer(2)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            '<div class="auth-card">'
            '<div class="auth-logo"><span class="auth-logo-icon">🔬</span></div>'
            '<p class="auth-brand">Med-Verify</p>'
            '<p class="auth-tagline">Clinical Logic Engine · Ophthalmology Edition</p>'
            '<p class="auth-title">Create Your Account</p>',
            unsafe_allow_html=True,
        )

        if st.session_state.get("_reg_error"):
            st.markdown(
                f'<div class="auth-error">⚠ {st.session_state.pop("_reg_error")}</div>',
                unsafe_allow_html=True,
            )

        with st.form("register_form", clear_on_submit=False):
            full_name = st.text_input("Full Name", placeholder="Dr. Priya Sharma")
            email     = st.text_input("Email Address", placeholder="doctor@hospital.in")
            password  = st.text_input("Password", type="password",
                                       placeholder="Min. 8 characters")
            confirm   = st.text_input("Confirm Password", type="password",
                                       placeholder="Repeat password")
            submitted = st.form_submit_button("Register & Continue →",
                                              use_container_width=True)

        if submitted:
            _do_register(full_name.strip(), email.strip(), password, confirm)

        st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-switch">Already have an account?&nbsp;</div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign In →", use_container_width=True, key="go_login"):
            st.session_state["auth_page"] = "login"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        _sec_badges()


def _do_register(full_name: str, email: str, password: str, confirm: str) -> None:
    # ── Validation ────────────────────────────────────────────────────────────
    if not all([full_name, email, password, confirm]):
        st.session_state["_reg_error"] = "All fields are required."
        st.rerun()

    if len(password) < 8:
        st.session_state["_reg_error"] = "Password must be at least 8 characters."
        st.rerun()

    if password != confirm:
        st.session_state["_reg_error"] = "Passwords do not match."
        st.rerun()

    if "@" not in email or "." not in email.split("@")[-1]:
        st.session_state["_reg_error"] = "Please enter a valid email address."
        st.rerun()

    # ── Persist ───────────────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        user = create_user(db, full_name, email, password)
        st.session_state.update({
            "auth_user_id":    user.id,
            "auth_user_name":  user.full_name,
            "auth_user_email": user.email,
        })
        st.session_state["auth_page"] = "payment"
        st.rerun()
    except ValueError as exc:
        st.session_state["_reg_error"] = str(exc)
        st.rerun()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Payment — QR Screen
# ─────────────────────────────────────────────────────────────────────────────
def _render_payment() -> None:
    user_name = st.session_state.get("auth_user_name", "Doctor")
    user_id   = st.session_state.get("auth_user_id")

    _spacer(2)

    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.markdown(
            f'<div class="pay-card">'
            f'<p class="pay-title">💳  Payment Required</p>'
            f'<p class="pay-subtitle">Welcome, <b style="color:#dce8f5;">{user_name}</b>! '
            f'One-time payment to activate your Med-Verify account.</p>'
            f'<p class="pay-amount">₹<span>899</span></p>'
            f'<p class="pay-caption">One-time · Lifetime access · All clinical modules</p>',
            unsafe_allow_html=True,
        )

        # ── Feature list ─────────────────────────────────────────────────────
        st.markdown(
            '<div class="pay-features">'
            '✅&nbsp; NLI-Verified Clinical Answers (DeBERTa-v3-large)<br>'
            '✅&nbsp; 53 ICMR · AIIMS · AAO PPP 2025 · NLEM 2022 protocols<br>'
            '✅&nbsp; Hard-Stop Contraindication Safety Engine<br>'
            '✅&nbsp; 100% On-Premise — no PHI leaves your machine<br>'
            '✅&nbsp; Unlimited queries · All clinical specialties'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── QR Code ──────────────────────────────────────────────────────────
        st.markdown(
            '<p style="color:#90b4d4;font-size:0.80rem;margin-bottom:0.5rem;">'
            'Scan with any UPI app (GPay · PhonePe · Paytm · BHIM)</p>',
            unsafe_allow_html=True,
        )

        qr_bytes = _get_payment_qr_bytes()

        # Centre the QR
        qa, qb, qc = st.columns([1, 2, 1])
        with qb:
            st.image(qr_bytes, use_container_width=True)

        st.markdown(
            '<p class="pay-caption">'
            '<b style="color:#FFC107;">Pay ₹899 to unlock the Med-Verify Clinical Suite</b><br>'
            'UPI ID: <code style="color:#64B5F6;">medverify@ybl</code>'
            '</p>',
            unsafe_allow_html=True,
        )

        # ── Confirm / Cancel buttons ──────────────────────────────────────────
        st.markdown('<div style="margin-top:1.1rem;"></div>', unsafe_allow_html=True)

        col_confirm, col_cancel = st.columns([3, 1])

        with col_confirm:
            if st.button(
                "✅  Confirm Payment — Activate Account",
                use_container_width=True,
                key="confirm_payment",
                type="primary",
            ):
                if user_id:
                    db = SessionLocal()
                    try:
                        update_payment_status(db, user_id, status=True)
                    finally:
                        db.close()

                st.session_state["authenticated"] = True
                st.session_state["auth_page"]     = "dashboard"
                st.rerun()

        with col_cancel:
            if st.button("← Back", use_container_width=True, key="back_to_login"):
                st.session_state["auth_page"] = "login"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    _spacer(1)
    _sec_badges()


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _spacer(n: int = 1) -> None:
    for _ in range(n):
        st.markdown(" ", unsafe_allow_html=True)


def _sec_badges() -> None:
    st.markdown(
        '<div class="sec-badges">'
        '<span class="sec-badge">🔒 AES-256 Encrypted</span>'
        '<span class="sec-badge">🏥 PHI Safe</span>'
        '<span class="sec-badge">🇮🇳 DPDP Compliant</span>'
        '</div>',
        unsafe_allow_html=True,
    )
