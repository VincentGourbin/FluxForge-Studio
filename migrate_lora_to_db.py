#!/usr/bin/env python3
"""
LoRA Migration Script
Migration script to transfer LoRA data from JSON file to SQLite database.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

def migrate_lora_to_database():
    """
    Migrate LoRA data from lora_info.json to SQLite database.
    Creates lora table if it doesn't exist and imports all LoRA data.
    """
    
    # Database file path
    db_path = 'generated_images.db'
    json_file = 'lora_info.json'
    
    print("🔄 Starting LoRA migration to database...")
    
    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"❌ Error: {json_file} not found!")
        return False
    
    # Load JSON data
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            lora_data = json.load(f)
        print(f"📂 Loaded {len(lora_data)} LoRA entries from {json_file}")
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create lora table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lora (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                activation_keyword TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                file_size INTEGER,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # Check if table was just created or already exists
        cursor.execute("SELECT COUNT(*) FROM lora")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"⚠️  Database already contains {existing_count} LoRA entries")
            response = input("Do you want to continue and potentially add duplicates? (y/n): ")
            if response.lower() != 'y':
                print("Migration cancelled by user")
                conn.close()
                return False
        
        # Migrate data
        migrated_count = 0
        skipped_count = 0
        current_time = datetime.now().isoformat()
        
        for lora_entry in lora_data:
            file_name = lora_entry.get('file_name', '')
            description = lora_entry.get('description', '')
            activation_keyword = lora_entry.get('activation_keyword', '')
            
            # Check if file exists in lora directory
            lora_file_path = Path('lora') / file_name
            file_size = None
            if lora_file_path.exists():
                file_size = lora_file_path.stat().st_size
            
            # Try to insert (skip if already exists)
            try:
                cursor.execute('''
                    INSERT INTO lora (file_name, description, activation_keyword, created_at, updated_at, file_size, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (file_name, description, activation_keyword, current_time, current_time, file_size, 1))
                migrated_count += 1
                
                # Status message
                status = "✅ Exists" if lora_file_path.exists() else "❌ Missing"
                print(f"  📁 {file_name} - {status}")
                
            except sqlite3.IntegrityError:
                # Entry already exists (duplicate file_name)
                skipped_count += 1
                print(f"  ⚠️  Skipped {file_name} (already exists)")
        
        # Commit changes
        conn.commit()
        
        print(f"\n✅ Migration completed:")
        print(f"  📊 {migrated_count} LoRA entries migrated")
        print(f"  ⏭️  {skipped_count} entries skipped (already existed)")
        print(f"  🗄️  Database: {db_path}")
        
        # Show summary
        cursor.execute("SELECT COUNT(*) FROM lora")
        total_count = cursor.fetchone()[0]
        print(f"  📈 Total LoRA entries in database: {total_count}")
        
        # Create backup of original JSON file
        backup_file = f"lora_info.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            import shutil
            shutil.copy2(json_file, backup_file)
            print(f"  💾 Backup created: {backup_file}")
        except Exception as e:
            print(f"  ⚠️  Could not create backup: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        conn.rollback()
        return False
    
    finally:
        conn.close()

def verify_migration():
    """
    Verify that the migration was successful by comparing JSON and database data.
    """
    print("\n🔍 Verifying migration...")
    
    # Load JSON data
    try:
        with open('lora_info.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        json_files = {entry['file_name'] for entry in json_data}
    except Exception as e:
        print(f"❌ Error loading JSON for verification: {e}")
        return False
    
    # Load database data
    try:
        conn = sqlite3.connect('generated_images.db')
        cursor = conn.cursor()
        cursor.execute("SELECT file_name FROM lora WHERE is_active = 1")
        db_files = {row[0] for row in cursor.fetchall()}
        conn.close()
    except Exception as e:
        print(f"❌ Error loading database for verification: {e}")
        return False
    
    # Compare
    missing_in_db = json_files - db_files
    extra_in_db = db_files - json_files
    
    if not missing_in_db and not extra_in_db:
        print("✅ Verification successful: JSON and database are in sync")
        return True
    else:
        if missing_in_db:
            print(f"⚠️  Missing in database: {missing_in_db}")
        if extra_in_db:
            print(f"⚠️  Extra in database: {extra_in_db}")
        return False

if __name__ == "__main__":
    print("🚀 LoRA Migration Tool")
    print("=" * 50)
    
    success = migrate_lora_to_database()
    
    if success:
        verify_migration()
        print("\n🎉 Migration process completed!")
        print("💡 You can now use the LoRA Management tab in the application")
    else:
        print("\n❌ Migration failed!")
        print("💡 Please check the error messages above and try again")