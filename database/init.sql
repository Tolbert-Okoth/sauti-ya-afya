-- 1. Enable UUID Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Create Counties Table (Geographic Backbone)
CREATE TABLE counties (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    geo_lat DECIMAL(9,6),
    geo_lng DECIMAL(9,6)
);

-- Seed Data: Insert Key Kenyan Counties
INSERT INTO counties (name, geo_lat, geo_lng) VALUES
('Nairobi', -1.2921, 36.8219),
('Mombasa', -4.0435, 39.6682),
('Kisumu', -0.0917, 34.7680),
('Turkana', 3.1167, 35.6000),
('Kiambu', -1.1714, 36.8356),
('Nakuru', -0.3031, 36.0800),
('Uasin Gishu', 0.5143, 35.2698),
('Kilifi', -3.6305, 39.8499),
('Machakos', -1.5177, 37.2634),
('Garissa', -0.4532, 39.6460)
ON CONFLICT (name) DO NOTHING;

-- 3. Create Users Table (Authentication)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    firebase_uid VARCHAR(128) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    role VARCHAR(20) CHECK (role IN ('CHW', 'DOCTOR', 'ADMIN')),
    county_id INT REFERENCES counties(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 4. Create Patients Table (NEW FOR PHASE 8)
-- Stores the real medical data captured by CHWs
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age VARCHAR(10),
    location VARCHAR(100),
    phone VARCHAR(20),
    
    -- Medical Data
    symptoms TEXT,
    diagnosis VARCHAR(255),      -- e.g. "High Freq Crackles"
    risk_level VARCHAR(50),      -- "High", "Medium", "Low"
    biomarkers JSONB,            -- Stores { zcr: 0.2, centroid: 3000 }
    
    -- Metadata
    audio_url TEXT,              -- URL to the file
    county_id INT REFERENCES counties(id), -- Links patient to a specific county
    created_at TIMESTAMP DEFAULT NOW()
);

-- 5. Enable Row Level Security (RLS)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;

-- 6. RLS Policies (The Security Rules)

-- Users can only see their own profile
CREATE POLICY user_isolation_policy ON users
    USING (firebase_uid = current_setting('app.current_user_uid', true));

-- Doctors can VIEW patients only in their own county
CREATE POLICY doctor_view_policy ON patients
    FOR SELECT
    USING (county_id = (
        SELECT county_id FROM users 
        WHERE firebase_uid = current_setting('app.current_user_uid', true)
    ));

-- CHWs can INSERT patients only into their own county
CREATE POLICY chw_insert_policy ON patients
    FOR INSERT
    WITH CHECK (county_id = (
        SELECT county_id FROM users 
        WHERE firebase_uid = current_setting('app.current_user_uid', true)
    ));

-- Doctors can DELETE (Resolve) patients in their own county
CREATE POLICY doctor_delete_policy ON patients
    FOR DELETE
    USING (county_id = (
        SELECT county_id FROM users 
        WHERE firebase_uid = current_setting('app.current_user_uid', true)
    ));