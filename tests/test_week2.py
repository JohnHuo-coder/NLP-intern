from scripts.text_cleaning import TextCleaner

def test_price_normalization():
    cleaner = TextCleaner()
    assert '450000' in cleaner.normalize_prices('priced at 450k')
    assert '1200000' in cleaner.normalize_prices('$1.2m home')
def test_profiling():
    cleaner = TextCleaner()
    profile = cleaner.profile_column(df, 'remarks')
    assert 'null_rate' in profile
    assert 'avg_length' in profile