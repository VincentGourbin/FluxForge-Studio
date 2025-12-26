#!/usr/bin/env python3
"""
FLUX.2 Database Migration Script

Migrates existing FLUX.1/Qwen/post-processing generation records to FLUX.2 format
while preserving all historical data and metadata.

Features:
- Automatic timestamped backup before migration
- Dry-run mode for testing without modifications
- Intelligent mapping of legacy generation types to FLUX.2 modes
- Backward compatibility with existing database schema
- Comprehensive error handling and validation

Usage:
    # Test migration (no changes)
    python migration_flux2.py --dry-run

    # Apply migration (with automatic backup)
    python migration_flux2.py

    # Apply without backup (dangerous, not recommended)
    python migration_flux2.py --no-backup

Author: FluxForge Team
License: MIT
"""

import sqlite3
import json
import argparse
import shutil
from datetime import datetime
from pathlib import Path


def backup_database(db_path):
    """Create timestamped backup of database.

    Args:
        db_path: Path to database file

    Returns:
        str: Path to backup file
    """
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path


def map_legacy_to_flux2_mode(generation_type, metadata):
    """Map legacy generation types to FLUX.2 modes.

    Args:
        generation_type: Original generation_type field value
        metadata: Metadata dictionary from JSON

    Returns:
        str: Corresponding FLUX.2 mode
    """

    # Direct mappings for most types
    mappings = {
        'standard': 'text-to-image',
        'kontext': 'image-to-image',
        'flux_redux': 'image-to-image',
        'flux_depth': 'depth-guided',
        'flux_canny': 'canny-guided',
        'qwen_generation': 'text-to-image',
        'controlnet': 'canny-guided',  # Assume canny controlnet
    }

    # Special handling for flux_fill (detect inpainting vs outpainting)
    if generation_type == 'flux_fill':
        mode = metadata.get('mode', 'Inpainting')
        return 'inpainting' if mode == 'Inpainting' else 'outpainting'

    return mappings.get(generation_type, 'text-to-image')


def migrate_record(record):
    """Migrate a single record to FLUX.2 format.

    Args:
        record: Database record tuple (id, timestamp, generation_type, output_filename, metadata_json)

    Returns:
        tuple: (new_metadata_json, record_id) or None if migration fails
    """
    record_id, timestamp, generation_type, output_filename, metadata_json = record

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as e:
        print(f"⚠️  Skipping record {record_id}: Invalid JSON - {e}")
        return None

    # Determine FLUX.2 mode
    flux2_mode = map_legacy_to_flux2_mode(generation_type, metadata)

    # Add new FLUX.2 fields
    metadata['flux2_mode'] = flux2_mode
    metadata['migrated_from'] = generation_type
    metadata['migration_date'] = datetime.now().isoformat()

    # Update model_alias for better filtering
    if 'model_alias' not in metadata:
        # No model_alias field - add based on generation type
        if generation_type == 'qwen_generation':
            metadata['model_alias'] = 'qwen-image (legacy)'
        else:
            metadata['model_alias'] = 'flux2-dev'
    elif metadata['model_alias'] in ['dev', 'krea-dev', 'schnell']:
        # FLUX.1 models → mark as FLUX.2 compatible
        if metadata['model_alias'] != 'schnell':  # Don't migrate schnell
            metadata['model_alias'] = 'flux2-dev'
    elif metadata['model_alias'] == 'qwen-image':
        # Mark Qwen as legacy
        metadata['model_alias'] = 'qwen-image (legacy)'

    # Ensure negative_prompt field exists (now available for all FLUX.2)
    if 'negative_prompt' not in metadata:
        metadata['negative_prompt'] = ''

    # Preserve quantization field if it exists
    if 'quantization' not in metadata:
        metadata['quantization'] = 'None'

    return (json.dumps(metadata), record_id)


def migrate_database(db_path, dry_run=False):
    """Migrate entire database to FLUX.2 format.

    Args:
        db_path: Path to database file
        dry_run: If True, perform migration without saving changes

    Returns:
        tuple: (migrated_count, failed_count)
    """

    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return 0, 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all records
    cursor.execute('SELECT id, timestamp, generation_type, output_filename, metadata_json FROM images')
    records = cursor.fetchall()

    print(f"📊 Found {len(records)} records to process")

    migrated_count = 0
    failed_count = 0
    skipped_count = 0

    for record in records:
        generation_type = record[2]  # generation_type field

        # Skip already migrated records
        if generation_type == 'flux2_generation':
            skipped_count += 1
            continue

        result = migrate_record(record)

        if result is None:
            failed_count += 1
            continue

        new_metadata_json, record_id = result

        if not dry_run:
            cursor.execute('''
                UPDATE images
                SET metadata_json = ?
                WHERE id = ?
            ''', (new_metadata_json, record_id))

        migrated_count += 1

        # Progress indicator
        if migrated_count % 100 == 0:
            print(f"  ... processed {migrated_count + failed_count + skipped_count}/{len(records)}")

    if not dry_run:
        conn.commit()

    conn.close()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}✅ Migration complete:")
    print(f"  ✓ Migrated: {migrated_count}")
    print(f"  ○ Skipped (already FLUX.2): {skipped_count}")
    print(f"  ✗ Failed: {failed_count}")

    return migrated_count, failed_count


def validate_migration(db_path):
    """Validate migration results.

    Args:
        db_path: Path to database file
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check for flux2_mode field in all records
    cursor.execute('SELECT id, metadata_json FROM images LIMIT 10')
    sample_records = cursor.fetchall()

    print("\n🔍 Validation Sample (first 10 records):")
    for record_id, metadata_json in sample_records:
        try:
            metadata = json.loads(metadata_json)
            flux2_mode = metadata.get('flux2_mode', 'MISSING')
            migrated_from = metadata.get('migrated_from', 'NONE')
            model_alias = metadata.get('model_alias', 'UNKNOWN')

            status = "✅" if flux2_mode != 'MISSING' else "⚠️"
            print(f"  {status} ID {record_id}: mode={flux2_mode}, from={migrated_from}, model={model_alias}")
        except json.JSONDecodeError:
            print(f"  ❌ ID {record_id}: Invalid JSON")

    # Statistics
    cursor.execute('SELECT generation_type, COUNT(*) FROM images GROUP BY generation_type')
    type_stats = cursor.fetchall()

    print("\n📈 Generation Type Statistics:")
    for gen_type, count in type_stats:
        print(f"  {gen_type}: {count} records")

    conn.close()


def main():
    """Main entry point for migration script."""

    parser = argparse.ArgumentParser(
        description="Migrate database to FLUX.2 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test migration without changes
  python migration_flux2.py --dry-run

  # Apply migration with backup
  python migration_flux2.py

  # Custom database path
  python migration_flux2.py --db /path/to/custom.db

  # Validate after migration
  python migration_flux2.py --validate
        """
    )

    parser.add_argument(
        '--db',
        default='generated_images.db',
        help='Path to database file (default: generated_images.db)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making changes (test mode)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup creation (not recommended)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate migration results without migrating'
    )

    args = parser.parse_args()

    db_path = args.db

    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return

    print("🚀 FLUX.2 Database Migration")
    print("=" * 60)

    # Validation mode
    if args.validate:
        validate_migration(db_path)
        return

    # Create backup
    if not args.no_backup and not args.dry_run:
        backup_path = backup_database(db_path)
        print(f"💾 Backup: {backup_path}\n")
    elif args.dry_run:
        print("🧪 DRY RUN MODE - No changes will be made\n")

    # Run migration
    migrated, failed = migrate_database(db_path, dry_run=args.dry_run)

    if args.dry_run:
        print("\n💡 This was a dry run. Use without --dry-run to apply changes.")
    else:
        print("\n✅ Migration applied successfully!")
        print("⚠️  Review History tab in app to verify data integrity")
        print("\n🔍 Run validation check:")
        print(f"  python migration_flux2.py --db {db_path} --validate")


if __name__ == "__main__":
    main()
