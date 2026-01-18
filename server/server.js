/* server/server.js */
const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const admin = require('firebase-admin');
const multer = require('multer');
const path = require('path');
const fs = require('fs'); // <--- Added fs to create folder safely
require('dotenv').config();

// Initialize Firebase
const serviceAccount = require('./config/serviceAccountKey.json');
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const app = express();

// ==========================================
// 1. SETUP UPLOADS FOLDER (Robust)
// ==========================================
// This ensures the server never crashes due to missing folder
const UPLOADS_DIR = path.join(__dirname, 'uploads');

const ensureUploadsFolder = () => {
    if (!fs.existsSync(UPLOADS_DIR)){
        console.log(`[System] Creating uploads folder at: ${UPLOADS_DIR}`);
        fs.mkdirSync(UPLOADS_DIR, { recursive: true });
    }
};
// Run on start
ensureUploadsFolder();

// ==========================================
// 2. MIDDLEWARE
// ==========================================
app.use(cors());

// Increase limit to 50mb to prevent PayloadTooLargeError
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// 🛑 ENABLE AUDIO PLAYBACK WITH CORS HEADERS
// This tells the browser: "It is safe to play this audio file from Port 5000"
app.use('/uploads', (req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS");
  res.header("Cross-Origin-Resource-Policy", "cross-origin"); // Critical for Audio
  next();
}, express.static(UPLOADS_DIR));

// ==========================================
// 3. DATABASE CONNECTION
// ==========================================
const db = new Pool({
  user: process.env.DB_USER || 'postgres',
  host: process.env.DB_HOST || 'localhost',
  database: process.env.DB_NAME || 'sautiyaafya',
  password: process.env.DB_PASSWORD || 'yourpassword',
  port: process.env.DB_PORT || 5432,
});

// Auto-Fix DB Column (Runs once to ensure column exists, then ignores)
(async () => {
  try {
    await db.query(`ALTER TABLE patients ADD COLUMN IF NOT EXISTS recording_url TEXT;`);
  } catch (err) {}
})();

// ==========================================
// 4. MULTER SETUP
// ==========================================
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    ensureUploadsFolder(); // Double check folder exists
    cb(null, UPLOADS_DIR); 
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, `rec-${uniqueSuffix}.webm`);
  }
});
const upload = multer({ storage });

// ==========================================
// 5. AUTH MIDDLEWARE
// ==========================================
const verifyToken = async (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).send('Unauthorized');
  try {
    const decodedValue = await admin.auth().verifyIdToken(token);
    req.user = decodedValue;
    next();
  } catch (e) {
    return res.status(403).send('Invalid Token');
  }
};

// ==========================================
// 6. ROUTES
// ==========================================

app.post('/api/login', verifyToken, async (req, res) => {
  const { email, uid } = req.user;
  try {
    let user = await db.query('SELECT * FROM users WHERE email = $1', [email]);
    if (user.rows.length === 0) {
      user = await db.query(
        "INSERT INTO users (email, role, firebase_uid, county_id) VALUES ($1, 'DOCTOR', $2, 1) RETURNING *",
        [email, uid]
      );
    }
    res.json({ role: user.rows[0].role });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/counties', verifyToken, async (req, res) => {
    try {
        const result = await db.query('SELECT * FROM counties ORDER BY code ASC');
        res.json(result.rows);
    } catch (err) {
        res.status(500).send(err.message);
    }
});

app.get('/api/system-config', verifyToken, async (req, res) => {
    res.json({ confidence_threshold: 0.75 });
});

app.post('/api/patients', verifyToken, upload.single('file'), async (req, res) => {
  try {
    const { 
        name, age, location, phone, symptoms, 
        diagnosis, risk_level, biomarkers, spectrogram 
    } = req.body;
    
    // Get filename from Multer
    const recordingUrl = req.file ? req.file.filename : null; 
    const email = req.user.email;

    // Get User Info
    const userRes = await db.query('SELECT county_id FROM users WHERE email = $1', [email]);
    if (userRes.rows.length === 0) return res.status(403).json({ message: "User invalid" });
    const { county_id } = userRes.rows[0];

    // Save to Database
    const query = `
      INSERT INTO patients 
      (name, age, location, phone, symptoms, diagnosis, risk_level, biomarkers, county_id, spectrogram, recording_url)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
      RETURNING *;
    `;
    
    const newPatient = await db.query(query, [
        name, age, location, phone, symptoms, diagnosis, 
        risk_level, biomarkers, county_id, spectrogram, recordingUrl
    ]);
    res.json(newPatient.rows[0]);

  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Failed to save patient record." });
  }
});

app.delete('/api/patients/:id', verifyToken, async (req, res) => {
    try {
        await db.query('DELETE FROM patients WHERE id = $1', [req.params.id]);
        res.json({ message: "Case Resolved" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Error deleting case" });
    }
});

app.get('/api/patients', verifyToken, async (req, res) => {
    try {
        const result = await db.query('SELECT * FROM patients ORDER BY created_at DESC');
        res.json(result.rows);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Admin & Analytics
app.get('/api/users', verifyToken, async (req, res) => {
  try {
    const result = await db.query('SELECT id, email, role, county_id, firebase_uid FROM users ORDER BY role');
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete('/api/users/:id', verifyToken, async (req, res) => {
  try {
    await db.query('DELETE FROM users WHERE id = $1', [req.params.id]);
    res.json({ message: "User deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/analytics', verifyToken, async (req, res) => {
  try {
    const riskQuery = await db.query('SELECT risk_level, COUNT(*) FROM patients GROUP BY risk_level');
    const locQuery = await db.query('SELECT location, COUNT(*) FROM patients GROUP BY location');
    res.json({ risk: riskQuery.rows, location: locQuery.rows });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Settings
app.get('/api/settings', verifyToken, async (req, res) => {
  try {
    const { email } = req.user;
    const userRes = await db.query('SELECT id FROM users WHERE email = $1', [email]);
    if (userRes.rows.length === 0) return res.status(404).send('User not found');
    const userId = userRes.rows[0].id;
    const settingsRes = await db.query('SELECT * FROM user_settings WHERE user_id = $1', [userId]);
    
    if (settingsRes.rows.length === 0) {
        const newSettings = await db.query(`INSERT INTO user_settings (user_id) VALUES ($1) RETURNING *`, [userId]);
        return res.json(newSettings.rows[0]);
    }
    res.json(settingsRes.rows[0]);
  } catch (err) {
    console.error(err);
    res.status(500).send('Server Error');
  }
});

app.put('/api/settings', verifyToken, async (req, res) => {
  try {
    const { email } = req.user;
    const { 
      language, offline_mode, auto_upload, notifications_enabled,
      confidence_threshold, export_moh, retain_logs 
    } = req.body;

    const userRes = await db.query('SELECT id FROM users WHERE email = $1', [email]);
    if (userRes.rows.length === 0) return res.status(404).send('User not found');
    const userId = userRes.rows[0].id;

    const updateRes = await db.query(
      `UPDATE user_settings 
       SET language = COALESCE($1, language),
           offline_mode = COALESCE($2, offline_mode),
           auto_upload = COALESCE($3, auto_upload),
           notifications_enabled = COALESCE($4, notifications_enabled),
           confidence_threshold = COALESCE($5, confidence_threshold),
           export_moh = COALESCE($6, export_moh),
           retain_logs = COALESCE($7, retain_logs),
           updated_at = CURRENT_TIMESTAMP
       WHERE user_id = $8
       RETURNING *`,
      [language, offline_mode, auto_upload, notifications_enabled, confidence_threshold, export_moh, retain_logs, userId]
    );
    res.json(updateRes.rows[0]);
  } catch (err) {
    console.error(err);
    res.status(500).send('Server Error');
  }
});

app.get('/', (req, res) => res.send("🛡️ SautiYaAfya Backend Secure & Online"));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Orchestrator secured and running on port ${PORT}`);
});