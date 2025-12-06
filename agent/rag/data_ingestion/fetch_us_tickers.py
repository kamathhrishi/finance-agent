#!/usr/bin/env python3
"""
Script to fetch all US equity tickers using financedatabase library
and optionally save them to a file or use them for earnings transcript fetching.
"""

import os
import sys
import re
import pandas as pd
from pathlib import Path
from typing import List, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('us_tickers_fetch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    import financedatabase as fd
except ImportError:
    logger.error("❌ financedatabase library not found. Please install it with: pip install financedatabase -U")
    sys.exit(1)

class USTickerFetcher:
    def __init__(self, output_file: str = "us_tickers.txt"):
        self.output_file = Path(output_file)
        
    def clean_ticker_name(self, ticker: str) -> bool:
        """
        Filter out tickers that have numbers in their names (usually bonds, etc.)
        Returns True if ticker should be kept, False if it should be filtered out
        """
        # Remove common suffixes that might contain numbers but are valid
        clean_ticker = re.sub(r'\.(A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z)$', '', ticker)
        
        # Check if ticker contains any digits
        if re.search(r'\d', clean_ticker):
            return False
            
        # Filter out common bond/derivative patterns
        bond_patterns = [
            r'\d+\.\d+%',  # Percentage bonds
            r'^\d+$',      # Pure numbers
            r'^\d+[A-Z]$', # Number followed by single letter
            r'^[A-Z]\d+$', # Letter followed by numbers
        ]
        
        for pattern in bond_patterns:
            if re.match(pattern, clean_ticker):
                return False
                
        return True
    
    def get_us_equity_tickers(self, 
                            only_primary_listing: bool = True,
                            exclude_otc: bool = True) -> List[str]:
        """
        Fetch all US equity tickers from financedatabase
        
        Args:
            only_primary_listing: Only get primary listings
            exclude_otc: Exclude OTC markets
        """
        try:
            logger.info("🔄 Initializing financedatabase...")
            equities = fd.Equities()
            
            logger.info("📊 Fetching US equity tickers...")
            
            # Build filters
            filters = {
                'country': 'United States',
                'only_primary_listing': only_primary_listing
            }
            
            # Fetch all US equities
            df = equities.select(**filters)
            
            if df.empty:
                logger.error("❌ No US equities found")
                return []
            
            logger.info(f"📈 Found {len(df)} total US equity listings")
            
            # Check available columns
            logger.info(f"📋 Available columns: {list(df.columns)}")
            
            # Filter out OTC if requested
            if exclude_otc:
                otc_markets = ['PNK', 'OTCQB', 'OTCQX', 'OTC Bulletin Board']
                df = df[~df['market'].isin(otc_markets)]
                logger.info(f"🚫 After excluding OTC markets: {len(df)} tickers")
            
            # In financedatabase, the ticker symbols are the dataframe index
            tickers = df.index.unique().tolist()
            logger.info(f"🎯 Found {len(tickers)} unique tickers from dataframe index")
            
            # Clean tickers (remove bonds, derivatives, etc.)
            logger.info("🧹 Cleaning ticker names...")
            clean_tickers = []
            removed_count = 0
            
            for ticker in tickers:
                if self.clean_ticker_name(ticker):
                    clean_tickers.append(ticker)
                else:
                    removed_count += 1
            
            logger.info(f"✅ Kept {len(clean_tickers)} clean tickers, removed {removed_count} bond/derivative tickers")
            
            # Sort tickers alphabetically
            clean_tickers.sort()
            
            return clean_tickers
            
        except Exception as e:
            logger.error(f"❌ Error fetching US equity tickers: {e}")
            return []
    
    def save_tickers_to_file(self, tickers: List[str]):
        """Save tickers to a text file"""
        try:
            with open(self.output_file, 'w') as f:
                for ticker in tickers:
                    f.write(f"{ticker}\n")
            
            logger.info(f"💾 Saved {len(tickers)} tickers to {self.output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving tickers to file: {e}")
    
    def save_tickers_to_csv(self, tickers: List[str], csv_file: str = "us_tickers_detailed.csv"):
        """Save tickers with additional details to CSV"""
        try:
            logger.info("🔄 Fetching detailed information for CSV...")
            equities = fd.Equities()
            
            # Get detailed info for each ticker
            detailed_data = []
            for ticker in tickers:
                try:
                    df = equities.select(country='United States', symbol=ticker)
                    if not df.empty:
                        detailed_data.append(df.iloc[0].to_dict())
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch details for {ticker}: {e}")
            
            if detailed_data:
                df_detailed = pd.DataFrame(detailed_data)
                df_detailed.to_csv(csv_file, index=False)
                logger.info(f"💾 Saved detailed info for {len(detailed_data)} tickers to {csv_file}")
            else:
                logger.warning("⚠️ No detailed data to save")
                
        except Exception as e:
            logger.error(f"❌ Error saving detailed CSV: {e}")
    
    def run(self, 
            only_primary_listing: bool = True,
            exclude_otc: bool = True,
            save_file: bool = True,
            save_csv: bool = False):
        """Main execution method"""
        
        logger.info("🚀 Starting US equity ticker fetch")
        logger.info(f"🏢 Primary listing only: {only_primary_listing}")
        logger.info(f"🚫 Exclude OTC: {exclude_otc}")
        
        # Fetch tickers
        tickers = self.get_us_equity_tickers(
            only_primary_listing=only_primary_listing,
            exclude_otc=exclude_otc
        )
        
        if not tickers:
            logger.error("❌ No tickers found")
            return []
        
        # Save to file if requested
        if save_file:
            self.save_tickers_to_file(tickers)
        
        # Save to CSV if requested
        if save_csv:
            self.save_tickers_to_csv(tickers)
        
        # Display sample of tickers
        logger.info("📋 Sample of fetched tickers:")
        for i, ticker in enumerate(tickers[:20]):
            logger.info(f"   {i+1:2d}. {ticker}")
        
        if len(tickers) > 20:
            logger.info(f"   ... and {len(tickers) - 20} more")
        
        logger.info(f"✅ Successfully fetched {len(tickers)} US equity tickers")
        return tickers

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch US equity tickers using financedatabase')
    parser.add_argument('--output-file', default='us_tickers.txt',
                       help='Output file for tickers')
    parser.add_argument('--include-otc', action='store_true',
                       help='Include OTC markets (default: exclude)')
    parser.add_argument('--include-secondary', action='store_true',
                       help='Include secondary listings (default: primary only)')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save detailed CSV with company information')
    parser.add_argument('--no-save-file', action='store_true',
                       help='Do not save tickers to text file')
    
    args = parser.parse_args()
    
    # Create fetcher
    fetcher = USTickerFetcher(output_file=args.output_file)
    
    try:
        tickers = fetcher.run(
            only_primary_listing=not args.include_secondary,
            exclude_otc=not args.include_otc,
            save_file=not args.no_save_file,
            save_csv=args.save_csv
        )
        
        print(f"\n🎉 Successfully fetched {len(tickers)} US equity tickers!")
        print(f"📁 Tickers saved to: {args.output_file}")
        
        if args.save_csv:
            print(f"📊 Detailed CSV saved to: us_tickers_detailed.csv")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Operation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
