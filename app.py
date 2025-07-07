import os
import sqlite3
import bcrypt
import logging
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
    g,
    make_response,
    flash,
)
from weasyprint import HTML
from werkzeug.utils import secure_filename
from datetime import datetime
from cachetools import TTLCache
from evaluator import evaluate_resume, get_resume_analyzer
import fitz
import asyncio
from markupsafe import escape
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")
app.config["UPLOAD_FOLDER"] = "Uploads"
app.config["DATABASE"] = os.path.join(app.root_path, "cvmanagizer.db")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Cache for database queries (TTL = 10 minutes)
db_cache = TTLCache(maxsize=100, ttl=600)


# Initialize database once at startup
def init_db():
    with app.app_context():
        db = sqlite3.connect(app.config["DATABASE"], check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cv_uploads (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                text TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS pro_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                job_title TEXT,
                keywords TEXT,
                sections TEXT,
                traits TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )
        db.commit()
        logger.info("Database initialized")
        db.close()


# Run database initialization once when the module is loaded
init_db()


# --- DB Helpers ---
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"], check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db:
        db.close()


def get_user_id():
    db = get_db()
    user = session.get("user")
    if user == "Guest":
        return "guest"
    if user:
        cache_key = f"user_id_{user}"
        if cache_key in db_cache:
            return db_cache[cache_key]
        row = db.execute("SELECT id FROM users WHERE username = ?", (user,)).fetchone()
        user_id = row["id"] if row else None
        db_cache[cache_key] = user_id
        return user_id
    return None


def log_activity(user_id, action):
    db = get_db()
    db.execute(
        "INSERT INTO activity_log (user_id, action) VALUES (?, ?)", (user_id, action)
    )
    db.commit()
    logger.info(f"Activity logged: {action} for user {user_id}")


def extract_text_from_pdf(filepath: str) -> str:
    try:
        doc = fitz.open(filepath)
        text = "\n".join(page.get_text() for page in doc)
        logger.info(f"Text extracted from {filepath}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {filepath}: {str(e)}")
        raise


def allowed_file(filename):
    return "." in filename and filename.lower().endswith(".pdf")


# --- Routes ---
@app.route("/", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def index():
    return redirect(url_for("welcome"))


@app.route("/welcome")
def welcome():
    return render_template("welcome.html", title="Welcome")


@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    db = get_db()
    user_id = get_user_id()
    # Pagination
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    per_page = 4
    offset = (page - 1) * per_page

    files = db.execute(
        "SELECT filename, uploaded_at FROM cv_uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT ? OFFSET ?",
        (user_id, per_page, offset)
    ).fetchall()

    total_files = db.execute(
        "SELECT COUNT(*) FROM cv_uploads WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    total_pages = (total_files + per_page - 1) // per_page

    # Helper object for template
    class Paginated:
        def __init__(self, items, page, per_page, total, total_pages):
            self.items = items
            self.page = page
            self.per_page = per_page
            self.total = total
            self.pages = total_pages
            self.has_prev = page > 1
            self.has_next = page < total_pages
            self.prev_num = page - 1
            self.next_num = page + 1

    files_paginated = Paginated(files, page, per_page, total_files, total_pages)
    log_activity(user_id, "Viewed index page with pagination")

    guest_limit_reached = session.pop("guest_limit_reached", None)
    return render_template(
        
        "index.html",
        title="Dashboard",
        files_paginated=files_paginated,
        enumerate=enumerate,
        guest_limit_reached=guest_limit_reached
    )



@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    files = request.files.getlist("file")
    job_description = request.form.get("job_description", "")

    if not files or not all(allowed_file(file.filename) for file in files):
        flash("Invalid file(s). Please upload PDF files.", "error")
        return redirect(url_for("home"))

    user_id = get_user_id()
    db = get_db()
    results = []

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        extracted = extract_text_from_pdf(filepath)

        job_title = "Software Engineer"
        use_custom_keywords = 0
        custom_keywords = ""
        if session["user"] != "Guest":
            setting = db.execute(
                "SELECT job_title, keywords, use_custom_keywords FROM pro_settings WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            ).fetchone()
            if setting:
                if setting["job_title"]:
                    job_title = setting["job_title"]
                # Use custom keywords only if user is pro and use_custom_keywords is enabled
                if session.get("pro_user") and setting["use_custom_keywords"]:
                    use_custom_keywords = setting["use_custom_keywords"]
                if use_custom_keywords and setting["keywords"]:
                    custom_keywords = setting["keywords"]

        from asgiref.sync import async_to_sync
        result = async_to_sync(evaluate_resume)(
            filepath,
            job_title=job_title,
            job_description=job_description,
            user_id=user_id,
            use_custom_keywords=use_custom_keywords,
            custom_keywords=custom_keywords,
        )
        results.append(result)

        if session["user"] == "Guest":
            if not session.get("guest_uploaded"):
                session["guest_uploaded"] = True
                session["evaluation"] = result
                log_activity("guest", "Uploaded resume")
                return redirect(url_for("results"))
            session["guest_limit_reached"] = True
            flash("Guest upload limit reached. Please sign up to continue.", "error")
            return redirect(url_for("home"))

        db.execute(
            "INSERT INTO cv_uploads (user_id, filename, text) VALUES (?, ?, ?)",
            (user_id, filename, extracted),
        )
        db.commit()
        log_activity(user_id, f"Uploaded resume: {filename}")

    session["evaluation"] = results[-1]  # Store the last result for display
    return redirect(url_for("results"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    filename = secure_filename(filename)
    log_activity(get_user_id(), f"Accessed uploaded file: {filename}")
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results")
def results():
    if "evaluation" not in session:
        return redirect(url_for("home"))
    result = session["evaluation"]
    log_activity(get_user_id(), "Viewed results")
    return render_template(
        "results.html",
        title="Results",
        finalscore=result["finalscore"],
        match_level=result["match_level"],
        strengths=result["strengths"],
        weaknesses=result["weaknesses"],
        suggestions=result["suggestions"],
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        db = get_db()
        email = escape(request.form["email"])
        password = request.form["password"].encode("utf-8")
        user = db.execute(
            "SELECT username, password FROM users WHERE email = ?", (email,)
        ).fetchone()
        if not user:
            logger.info(f"Login failed: Email {email} not found")
            flash("Email not found", "error")
            return render_template("login.html", title="Login")
        try:
            if bcrypt.checkpw(password, user["password"].encode("utf-8")):
                session["user"] = user["username"]
                log_activity(get_user_id(), "Logged in")
                return redirect(url_for("index_page"))
            else:
                logger.info(f"Login failed: Incorrect password for {email}")
                flash("Incorrect password", "error")
        except Exception as e:
            logger.error(f"Login failed for {email}: Invalid password hash - {str(e)}")
            flash(
                "Your account needs a password reset. Please reset your password.",
                "error",
            )
            return redirect(url_for("reset_password"))
        return render_template("login.html", title="Login")
    return render_template("login.html", title="Login")


@app.route("/index_page")
def index_page():
    return redirect(url_for("home"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        db = get_db()
        try:
            username = escape(request.form["username"])
            email = escape(request.form["email"])
            password = request.form["password"].encode("utf-8")
        except KeyError as e:
            logger.error(f"Signup failed: Missing form field {str(e)}")
            flash("Please fill out all required fields", "error")
            return render_template("signup.html", title="Sign Up")
        exists = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if exists:
            logger.info(f"Signup failed: Email {email} already in use")
            flash("Email already in use", "error")
            return render_template("signup.html", title="Sign Up")
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt()).decode("utf-8")
        try:
            logger.info(f"Attempting to insert user {username} with email {email}")
            db.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed_password),
            )
            db.commit()
            session["user"] = username
            log_activity(get_user_id(), "Signed up")
            logger.info(f"Signup successful for {email}")
            return redirect(url_for("home"))
        except sqlite3.Error as e:
            logger.error(f"Signup failed for {email}: Database error - {str(e)}")
            flash("An error occurred. Please try again.", "error")
            return render_template("signup.html", title="Sign Up")
    return render_template("signup.html", title="Sign Up")


@app.route("/logout")
def logout():
    user_id = get_user_id()
    session.clear()
    log_activity(user_id, "Logged out")
    return redirect(url_for("welcome"))


@app.route("/guest")
def guest():
    session["user"] = "Guest"
    log_activity("guest", "Accessed as guest")
    return redirect(url_for("home"))


@app.route("/settings")
def settings():
    if "user" not in session or session["user"] == "Guest":
        return render_template("settings.html", title="Settings")
    user_id = get_user_id()
    db = get_db()
    saved = db.execute(
        "SELECT * FROM pro_settings WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    log_activity(user_id, "Viewed settings")
    return render_template("settings.html", title="Settings", saved=saved)


@app.route("/delete-account", methods=["POST"])
def delete_account():
    if "user" not in session:
        return redirect(url_for("login"))
    user_id = get_user_id()
    db = get_db()
    db.execute("DELETE FROM cv_uploads WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM pro_settings WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM activity_log WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    session.clear()
    log_activity(user_id, "Deleted account")
    return redirect(url_for("welcome"))


@app.route("/pro")
def pro():
    if "user" not in session:
        return redirect(url_for("login"))

    if not session.get("pro_user") and not request.args.get("activated"):
        flash("promo_block", "promo")  # ✅ Must be indented

    user_id = get_user_id()
    db = get_db()
    saved = db.execute(
        "SELECT * FROM pro_settings WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    log_activity(user_id, "Viewed pro settings")
    return render_template("pro.html", title="Go Pro", success=False, saved=saved)


    user_id = get_user_id()
    db = get_db()
    saved = db.execute(
        "SELECT * FROM pro_settings WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    log_activity(user_id, "Viewed pro settings")
    return render_template("pro.html", title="Go Pro", success=False, saved=saved)

@app.route("/redeem-promo", methods=["POST"])
def redeem_promo():
    if "user" not in session:
        return redirect(url_for("login"))

    code_entered = request.form.get("promo_code", "").strip().upper()

    VALID_PROMO_CODES = {
        "PRO2024",
        "BOOSTAI",
        "UNLOCK50",
        "CVWIZARD",
        "TOPTIER"
    }

    if code_entered in VALID_PROMO_CODES:
        session["pro_user"] = True
        flash("⚡ Pro Activated Successfully!", "success")
        log_activity(get_user_id(), f"Activated Pro with promo code: {code_entered}")
    else:
        flash("❌ Invalid promo code. Please try again.", "error")

    return redirect(url_for("pro", activated=1))





@app.route("/submit-pro-settings", methods=["POST"])
def submit_pro_settings():
    if "user" not in session:
        return redirect(url_for("login"))

    if not session.get("pro_user"):
        flash("You must activate Pro with a promo code to save Pro settings.", "error")
        return redirect(url_for("pro"))

    user_id = get_user_id()
    db = get_db()
    job_title = escape(request.form.get("job_title", ""))
    keywords = escape(request.form.get("keywords", ""))
    use_custom_keywords = 1 if request.form.get("use_custom_keywords") == "1" else 0
    sections = escape(request.form.get("sections", ""))
    traits = escape(request.form.get("traits", ""))
    notes = escape(request.form.get("notes", ""))
    db.execute(
        """
        INSERT INTO pro_settings (user_id, job_title, keywords, use_custom_keywords, sections, traits, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, job_title, keywords, use_custom_keywords, sections, traits, notes),
    )
    db.commit()
    # Update session pro_user status based on saved settings
    if use_custom_keywords == 1 or job_title or keywords or sections or traits or notes:
        session["pro_user"] = True
    else:
        session.pop("pro_user", None)
    log_activity(user_id, "Submitted pro settings")
    flash("Pro settings saved successfully.", "success")
    return redirect(url_for("home"))



@app.route("/reset-pro-settings", methods=["POST"])
def reset_pro_settings():
    if "user" not in session:
        return redirect(url_for("login"))
    user_id = get_user_id()
    db = get_db()
    db.execute("DELETE FROM pro_settings WHERE user_id = ?", (user_id,))
    db.commit()
    log_activity(user_id, "Reset pro settings")
    flash("Pro settings have been reset.", "info")
    return redirect(url_for("settings"))

@app.route("/clear_upload_history", methods=["POST"])
def clear_upload_history():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = get_user_id()
    db = get_db()
    db.execute("DELETE FROM cv_uploads WHERE user_id = ?", (user_id,))
    db.commit()
    log_activity(user_id, "Cleared upload history")
    flash("Upload history cleared successfully.", "success")
    return redirect(url_for("settings"))



@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        db = get_db()
        email = escape(request.form["email"])
        new_password = request.form["password"].encode("utf-8")
        hashed_password = bcrypt.hashpw(new_password, bcrypt.gensalt()).decode("utf-8")
        db.execute(
            "UPDATE users SET password = ? WHERE email = ?", (hashed_password, email)
        )
        db.commit()
        flash("Password has been reset. You can now log in.", "success")
        return redirect(url_for("login"))
    return render_template("reset_password.html", title="Reset Password")


@app.route("/about")
def about():
    log_activity(get_user_id(), "Viewed about page")
    return render_template("about.html", title="About Us")

# --- ADD to app.py ---

@app.route("/change-password", methods=["POST"])
def change_password():
    if "user" not in session:
        return redirect(url_for("login"))

    db = get_db()
    user_id = get_user_id()
    user = db.execute("SELECT password FROM users WHERE id = ?", (user_id,)).fetchone()

    old_password = request.form.get("old_password", "").encode("utf-8")
    new_password = request.form.get("new_password", "").encode("utf-8")
    confirm_password = request.form.get("confirm_password", "").encode("utf-8")

    if not bcrypt.checkpw(old_password, user["password"].encode("utf-8")):
        flash("Incorrect current password.", "error")
        return redirect(url_for("settings"))

    if new_password != confirm_password:
        flash("New passwords do not match.", "error")
        return redirect(url_for("settings"))

    hashed_new = bcrypt.hashpw(new_password, bcrypt.gensalt()).decode("utf-8")
    db.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_new, user_id))
    db.commit()
    flash("Password updated successfully.", "success")
    log_activity(user_id, "Changed password")
    return redirect(url_for("settings"))


@app.route("/toggle-pro-default", methods=["POST"])
def toggle_pro_default():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = get_user_id()
    db = get_db()
    default_mode = request.form.get("default_mode") == "evaluator"
    use_custom = 0 if default_mode else 1

    db.execute("""
        UPDATE pro_settings SET use_custom_keywords = ? 
        WHERE user_id = ? AND id = (
            SELECT id FROM pro_settings WHERE user_id = ? ORDER BY created_at DESC LIMIT 1
        )
    """, (use_custom, user_id, user_id))
    db.commit()
    flash("Pro settings mode updated.", "info")
    log_activity(user_id, "Toggled pro default mode")
    return redirect(url_for("settings"))



@app.route("/export-pdf")
def export_pdf():
    if "user" not in session:
        return redirect(url_for("login"))

    username = escape(session.get("user", "User"))
    now = datetime.now().strftime("%Y-%m-%d")
    result = session.get("evaluation", {})
    finalscore = result.get("finalscore", 0)
    match_level = result.get("match_level", "N/A")
    strengths = result.get("strengths", [])
    weaknesses = result.get("weaknesses", [])
    suggestions = result.get("suggestions", [])

    html_content = render_template(
        "export_template.html",
        username=username,
        now=now,
        finalscore=finalscore,
        match_level=match_level,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
    )

    try:
        pdf = HTML(string=html_content, base_url=request.base_url).write_pdf()
        response = make_response(pdf)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = (
            f"inline; filename=CV_Analysis_{username}_{now}.pdf"
        )
        log_activity(get_user_id(), "Exported PDF report")
        return response
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        flash("Failed to generate PDF. Please try again.", "error")
        return redirect(url_for("results"))

@app.route("/toggle_pro_off", methods=["POST"])
def toggle_pro_off():
    if "user" not in session:
        return redirect(url_for("login"))

    if "deactivate_pro" in request.form:
        session.pop("pro_user", None)
        flash("Pro mode has been disabled.", "info")
    else:
        session["pro_user"] = True
        flash("Pro mode reactivated.", "success")

    return redirect(url_for("settings"))

@app.route("/clear-pro-criteria", methods=["POST"])
def clear_pro_criteria():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = get_user_id()
    db = get_db()
    db.execute("DELETE FROM pro_settings WHERE user_id = ?", (user_id,))
    db.commit()
    flash("Saved Pro evaluation criteria deleted successfully.", "info")
    log_activity(user_id, "Cleared saved pro criteria")
    return redirect(url_for("settings"))


@app.route("/clear-guest-flag")
def clear_guest_flag():
    session.pop("guest_limit_reached", None)
    session.pop("guest_uploaded", None)
    log_activity("guest", "Cleared guest flag")
    return "", 204


@app.route("/api/uploads")
def api_uploads():
    if "user" not in session:
        return {"error": "Unauthorized"}, 401
    db = get_db()
    user_id = get_user_id()
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    per_page = 4
    offset = (page - 1) * per_page

    files = db.execute(
        "SELECT filename, uploaded_at FROM cv_uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT ? OFFSET ?",
        (user_id, per_page, offset)
    ).fetchall()

    total_files = db.execute(
        "SELECT COUNT(*) FROM cv_uploads WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    total_pages = (total_files + per_page - 1) // per_page

    files_list = [{"filename": f["filename"], "uploaded_at": f["uploaded_at"]} for f in files]

    return {
        "files": files_list,
        "page": page,
        "per_page": per_page,
        "total_files": total_files,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_num": page - 1,
        "next_num": page + 1,
    }


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
