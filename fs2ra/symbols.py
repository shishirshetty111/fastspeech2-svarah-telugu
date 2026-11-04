"""Extend the upstream symbol table with a language's phone set.

Upstream maps "{p1 p2}" phone strings through text._symbol_to_id with "@"+phone and
silently drops unknown symbols; the encoder embedding size is len(text.symbols.symbols)+1
read at model construction. Telugu phones are appended in place (phones whose "@"
symbol already exists, e.g. pinyin initials "@k", keep that id) before the model is
built, so English (no extra phones) keeps the upstream 360 symbols (361 embeddings).

Must also run inside DataLoader workers (worker_init_fn) where processes are spawned.
"""

LANGUAGE_PHONES = {"english": [], "telugu": None}  # telugu filled lazily from fs2ra.text_te


def phones_for(language):
    if language == "telugu":
        from fs2ra.text_te import PHONES

        return list(PHONES)
    return list(LANGUAGE_PHONES.get(language) or [])


def register_phones(phones):
    import text  # upstream
    from text.symbols import symbols

    for p in phones:
        sym = "@" + p
        if sym not in text._symbol_to_id:
            text._symbol_to_id[sym] = len(symbols)
            text._id_to_symbol[len(symbols)] = sym
            symbols.append(sym)
    return len(symbols)


def register_language(language):
    return register_phones(phones_for(language))


def check_phone_strings(lines, language):
    """Every phone in each '{...}' string must map to one id (upstream drops unknowns)."""
    from text import text_to_sequence

    for phones in lines:
        n = len(phones.strip("{}").split())
        ids = text_to_sequence(phones, [])
        if len(ids) != n:
            raise ValueError("%s: %d of %d phones unknown to the symbol table in %s"
                             % (language, n - len(ids), n, phones))
