#!/usr/bin/env python3
"""
SkillCorner Data Download Script

Downloads match data from SkillCorner API for the MLS 2023 season.

Usage:
    1. Add your credentials in the section below (or use .env file)

    2. Run the script:
       python3 download_skillcorner_data.py

    3. Optionally specify how many matches to download:
       python3 download_skillcorner_data.py --limit 10

The script will:
- Download match metadata (JSON)
- Download dynamic events (CSV)
- Download tracking data (JSONL)
- Skip files that already exist
- Resume from where it left off if interrupted

Data is saved to: ./skillcorner_download/
"""

# ============================================================================
# CREDENTIALS - Add your SkillCorner credentials here
# ============================================================================
SKILLCORNER_USERNAME = "jlocala1@jh.edu"
SKILLCORNER_PASSWORD = "lMy6UqkA3FtN5Bzp"
# ============================================================================

import sys
import argparse
import requests
from pathlib import Path


def download_skillcorner_data(output_dir='skillcorner_download',
                               fetch_per_batch=5,
                               total_batches=5,
                               competition_edition=419):
    """
    Download SkillCorner match data from the API.

    Args:
        output_dir: Directory to save downloaded files
        fetch_per_batch: Number of matches to fetch per API call
        total_batches: Number of batches to download (total matches = fetch_per_batch * total_batches)
        competition_edition: Competition edition ID (419 = MLS 2023)
    """

    # Load credentials from top of file
    username = SKILLCORNER_USERNAME
    password = SKILLCORNER_PASSWORD

    if not username or not password:
        raise SystemExit(
            'ERROR: Missing SkillCorner credentials.\n\n'
            'Edit this script and add credentials at the top (lines 29-30):\n'
            '  SKILLCORNER_USERNAME = "your_email@example.com"\n'
            '  SKILLCORNER_PASSWORD = "your_password"\n'
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")

    # Setup authenticated session
    session = requests.Session()
    session.auth = (username, password)

    # Check what matches we already have
    existing_matches = set(
        p.stem.split('_')[0]
        for p in output_path.glob('*_dynamic_events.csv')
    )
    print(f'Already have {len(existing_matches)} matches downloaded')

    # Start offset based on what we already have
    offset = len(existing_matches)

    def save_file(url, file_path, headers=None, stream=False):
        """
        Download and save a file from URL.

        Returns:
            str: Status message ('skip', 'ok', 'empty', or 'fail ...')
        """
        if file_path.exists():
            return 'skip'

        try:
            response = session.get(
                url,
                headers=headers,
                stream=stream,
                timeout=(30, 300)  # (connect, read) timeouts
            )

            if response.status_code == 204:
                return 'empty'

            response.raise_for_status()

            if stream:
                # For large files (tracking data), stream in chunks
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=64*1024):
                        if chunk:
                            f.write(chunk)
            else:
                # For small files, write all at once
                with open(file_path, 'wb') as f:
                    f.write(response.content)

            return 'ok'

        except Exception as e:
            # Clean up partial file on error
            if file_path.exists():
                file_path.unlink()
            return f'fail {e}'

    # Download matches in batches
    for batch_num in range(total_batches):
        print(f"\n--- Batch {batch_num + 1}/{total_batches} ---")

        # Fetch list of matches
        try:
            response = session.get(
                'https://skillcorner.com/api/matches',
                params={
                    'competition_edition': competition_edition,
                    'limit': fetch_per_batch,
                    'offset': offset
                },
                timeout=60
            )
            response.raise_for_status()
            results = response.json().get('results', [])
        except Exception as e:
            print(f'ERROR fetching match list: {e}')
            break

        if not results:
            print('No more matches available; stopping')
            break

        offset += len(results)

        # Download data for each match
        for match in results:
            match_id = str(match['id'])

            if match_id in existing_matches:
                print(f'Match {match_id}: already downloaded, skipping')
                continue

            print(f'Match {match_id}:')

            # Download match metadata (JSON)
            status = save_file(
                f'https://skillcorner.com/api/match/{match_id}',
                output_path / f'match_{match_id}.json'
            )
            print(f'  match metadata: {status}')

            # Download events (CSV)
            status = save_file(
                f'https://skillcorner.com/api/match/{match_id}/dynamic_events',
                output_path / f'{match_id}_dynamic_events.csv',
                headers={'Accept': 'text/csv'}
            )
            print(f'  events: {status}')

            # Download tracking data (JSONL) - streamed due to size
            status = save_file(
                f'https://skillcorner.com/api/match/{match_id}/tracking',
                output_path / f'{match_id}_tracking_extrapolated.jsonl',
                stream=True
            )
            print(f'  tracking: {status}')

            existing_matches.add(match_id)

        print(f'Progress: {len(existing_matches)} total matches, next offset: {offset}')

    print(f'\n=== Download Complete ===')
    print(f'Total matches downloaded: {len(existing_matches)}')
    print(f'Data saved to: {output_path.absolute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download SkillCorner match data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--output-dir',
        default='skillcorner_download',
        help='Directory to save downloaded files (default: skillcorner_download)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of matches to download (default: 25 = 5 batches × 5 matches)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of matches to fetch per API call (default: 5)'
    )
    parser.add_argument(
        '--competition',
        type=int,
        default=419,
        help='Competition edition ID (default: 419 for MLS 2023)'
    )

    args = parser.parse_args()

    # Calculate total_batches from limit if provided
    if args.limit:
        total_batches = (args.limit + args.batch_size - 1) // args.batch_size
    else:
        total_batches = 5  # Default: 5 batches × 5 matches = 25 matches

    print("SkillCorner Data Download Script")
    print("=" * 50)
    print(f"Batch size: {args.batch_size} matches")
    print(f"Total batches: {total_batches}")
    print(f"Max matches to download: {args.batch_size * total_batches}")
    print(f"Competition edition: {args.competition}")
    print("=" * 50)

    try:
        download_skillcorner_data(
            output_dir=args.output_dir,
            fetch_per_batch=args.batch_size,
            total_batches=total_batches,
            competition_edition=args.competition
        )
    except KeyboardInterrupt:
        print('\n\nDownload interrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f'\n\nERROR: {e}')
        sys.exit(1)
