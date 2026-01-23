/* server/server.js */
const express = require('express');
const cors = require('cors'); // ✅ Use the standard package
const { Pool } = require('pg');
const admin = require('firebase-admin'); 
const multer = require('multer');
const cloudinary = require('cloudinary').v2; 
const stream = require('stream'); 
require('dotenv').config();

const app = express();

// ==========================================
// 1. CONFIGURATION
// ==========================================

cloudinary.config({ 
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME, 
  api_key: process.env.CLOUDINARY_API_KEY, 
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true
});

let serviceAccount;
try {
  if (process.env.FIREBASE_SERVICE_ACCOUNT) {
    serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
  } else {
    serviceAccount = require('./config/serviceAccountKey.json');
  }

  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
  });
} catch (error) {
  console.error("Firebase Init Error:", error.message);
}

// ==========================================
// 2. MIDDLEWARE (🛡️ STANDARD CORS SECURITY)
// ==========================================

const allowedOrigins = [
  "http://localhost:3000",                                      
  "https://sautiyaafya.vercel.app",                             
  "https://sauti-ya-afya-git-main-tolberts-projects.vercel.app" 
];

const corsOptions = {
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (allowedOrigins.indexOf(origin) !== -1) {
      return callback(null, true);
    } else {
      console.log(`Blocked by CORS: ${origin}`);
      return callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true, 
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
};

app.use(cors(corsOptions));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// ==========================================
// 3. DATABASE CONNECTION
// ==========================================
const dbConfig = process.env.DATABASE_URL 
  ? {
      connectionString: process.env.DATABASE_URL,
      ssl: { rejectUnauthorized: false } 
    }
  : {
      user: process.env.DB_USER || 'postgres',
      host: process.env.DB_HOST || 'localhost',
      database: process.env.DB_NAME || 'sautiyaafya',
      password: process.env.DB_PASSWORD || 'yourpassword',
      port: process.env.DB_PORT || 5432,
      ssl: false
    };

const db = new Pool(dbConfig);

// ==========================================
// 4. MULTER & CLOUDINARY
// ==========================================
const storage = multer.memoryStorage();
const upload = multer({ storage });

const uploadToCloudinary = (buffer, folder = 'audio') => {
  return new Promise((resolve, reject) => {
    const uploadStream = cloudinary.uploader.upload_stream(
      { resource_type: "auto", folder: folder },
      (error, result) => {
        if (error) return reject(error);
        resolve(result);
      }
    );
    const bufferStream = new stream.PassThrough();
    bufferStream.end(buffer);
    bufferStream.pipe(uploadStream);
  });
};

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

app.get('/', (req, res) => res.send("🛡️ SautiYaAfya Standardized Backend Online"));
app.get('/health', (req, res) => res.status(200).send('OK'));

// 🛠️ DATABASE SETUP ROUTE (RUN THIS ONCE)
app.get('/api/init-db', async (req, res) => {
  try {
    // 1. Create Users Table
    await db.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        firebase_uid VARCHAR(255) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        role VARCHAR(50) DEFAULT 'CHW',
        county_id INTEGER DEFAULT 47,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    // 2. Create Patients Table
    await db.query(`
      CREATE TABLE IF NOT EXISTS patients (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255),
        age INTEGER,
        location VARCHAR(255),
        phone VARCHAR(50),
        symptoms TEXT,
        diagnosis TEXT,
        risk_level VARCHAR(50),
        biomarkers JSONB,
        spectrogram TEXT,
        recording_url TEXT,
        county_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    // 3. Create Settings Table
    await db.query(`
      CREATE TABLE IF NOT EXISTS user_settings (
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        language VARCHAR(10) DEFAULT 'en',
        offline_mode BOOLEAN DEFAULT false,
        auto_upload BOOLEAN DEFAULT true,
        notifications_enabled BOOLEAN DEFAULT true,
        confidence_threshold DECIMAL DEFAULT 0.75,
        export_moh BOOLEAN DEFAULT false,
        retain_logs BOOLEAN DEFAULT true,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id)
      );
    `);

     // 4. Create Counties Table
     await db.query(`
        CREATE TABLE IF NOT EXISTS counties (
            id SERIAL PRIMARY KEY,
            code INTEGER,
            name VARCHAR(255)
        );
     `);

    res.send("✅ Database tables created successfully!");
  } catch (err) {
    console.error(err);
    res.status(500).send("❌ Database creation failed: " + err.message);
  }
});

app.post('/api/login', verifyToken, async (req, res) => {
  const { email, uid } = req.user;
  try {
    let user = await db.query('SELECT * FROM users WHERE email = $1', [email]);
    if (user.rows.length === 0) {
      user = await db.query(
        "INSERT INTO users (id, email, role, firebase_uid, created_at) VALUES ($1, $2, 'doctor', $3, NOW()) RETURNING *",
        [uid, email, uid]
      );
    }
    res.json({ role: user.rows[0].role });
  } catch (err) {
    console.error("Login DB Error:", err);
    res.status(500).json({ error: err.message });
  }
});

// ... (Rest of your routes stay the same as your previous file) ...
// Ensure register, patients, etc. are below here.

// NOTE: For the sake of space in this reply, I'm assuming you will keep 
// the rest of the routes (register, patients, analytics, settings) 
// exactly as they were in your previous paste. They were correct.
// Make sure you don't delete them!

app.post('/api/register', verifyToken, async (req, res) => {
    const { role, county_id } = req.body;
    const { email, uid } = req.user;
    try {
        const newUser = await db.query(
            "INSERT INTO users (firebase_uid, email, role, county_id) VALUES ($1, $2, $3, $4) RETURNING *",
            [uid, email, role || 'CHW', county_id || 1]
        );
        res.json(newUser.rows[0]);
    } catch (err) {
        console.error("Registration Error:", err);
        res.status(500).json({ message: "Server Error" });
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
    
    let recordingUrl = null;

    if (req.file) {
        console.log("Uploading to Cloudinary...");
        const result = await uploadToCloudinary(req.file.buffer, 'sautiyaafya-audio');
        recordingUrl = result.secure_url; 
        console.log(`[Cloud] Uploaded: ${recordingUrl}`);
    }

    const email = req.user.email;
    const userRes = await db.query('SELECT county_id FROM users WHERE email = $1', [email]);
    const county_id = (userRes.rows.length > 0) ? userRes.rows[0].county_id : 47; 

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
    console.error("Upload Error:", err);
    res.status(500).json({ message: "Failed to save patient record." });
  }
});

app.delete('/api/patients/:id', verifyToken, async (req, res) => {
    try {
        await db.query('DELETE FROM patients WHERE id = $1', [req.params.id]);
        res.json({ message: "Case Resolved" });
    } catch (err) {
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

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Orchestrator secured and running on port ${PORT}`);
});