import logging

class SymbolFormatter(logging.Formatter):
    SYMBOLS = {
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❗",
        logging.DEBUG: "🐞",
        logging.CRITICAL: "💥"
    }

    def format(self, record):
        symbol = self.SYMBOLS.get(record.levelno, "❓")
        record.symbol = symbol
        return f"{record.symbol} {super().format(record)}"

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File Handler
    fh = logging.FileHandler('logs/error.log')
    fh.setLevel(logging.ERROR)

    # Formatter with Symbols
    formatter = SymbolFormatter('%(asctime)s - %(symbol)1s %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger