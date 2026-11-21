"""Channel niches: system prompt + defaults for Groq script generation.

Each preset includes a topic_pool — a list of setting/situation ideas.
One is picked randomly per run if no --topic is provided, ensuring variety.
"""

from __future__ import annotations

from typing import TypedDict


class Variant(TypedDict, total=False):
    """One output variant — same images, different audio/subs/upload target."""
    lang: str  # "en", "hi", etc. used as key in Groq response
    label: str  # human-readable for logs
    tts_voice: str  # Edge TTS voice (e.g. "hi-IN-MadhurNeural")
    caption_font: str  # font filename inside assets/fonts/
    caption_font_name: str  # FFmpeg-visible font family name
    yt_token_env: str  # env var name for YouTube refresh token (e.g. "YT_REFRESH_TOKEN_HI")
    min_words: int  # min word count for narration validation


class ChannelPreset(TypedDict, total=False):
    id: str
    label: str
    groq_system_hint: str
    segment_count: int  # images + script beats
    topic_pool: list[str]
    image_style_suffix: str  # appended to every image prompt
    image_negative_prompt: str  # passed as negative prompt
    # Single-variant fields (backward compat — used when `variants` is absent):
    language: str
    tts_voice: str
    caption_font: str
    caption_font_name: str
    min_words: int  # min word count for narration validation (single-variant)
    # Multi-variant mode — Groq returns translations for each lang, pipeline renders+uploads per variant.
    variants: list[Variant]
    # topic_rotation: "myth" → pipeline/myth_topics.py (IST day theme + no-repeat within theme)
    topic_rotation: str
    # Single-variant YouTube upload: which env var holds this channel's refresh token
    yt_token_env: str
    # Extra uploads: same MP4 uploaded to additional channels using these env var names
    extra_yt_token_envs: list[str]


PRESETS: dict[str, ChannelPreset] = {
    "facts": {
        "id": "facts",
        "label": "Mind-blowing facts Short (English)",
        "variants": [
            {
                "lang": "en",
                "label": "English",
                "tts_voice": "en-US-GuyNeural",
                "caption_font": "BebasNeue-Regular.ttf",
                "caption_font_name": "Bebas Neue",
                "yt_token_env": "YT_REFRESH_TOKEN_EN",
                "min_words": 70,
            },
        ],
        "groq_system_hint": (
            "You write punchy YouTube Shorts about surprising, verified facts in English. "
            "STRUCTURE: hook fact in opening, supporting facts in the middle, punchline + takeaway at end. "
            "TONE: energetic, curious, confident. No clickbait lies. "
            "Each fact must be broadly accurate; if unsure, use safer wording like "
            "'scientists believe' or 'some research suggests'. "
            "No medical advice. No hashtags inside narration. Original phrasing only. "
            "IMAGE PROMPT RULE: write image prompts in ENGLISH only. Describe real photographs or documentary stills. "
            "Use real-world subjects, real lighting, real environments. NEVER write 'cartoon', 'illustration', "
            "'anime', or 'stylized'. Examples: 'a real octopus underwater in clear blue ocean, sunlight rays', "
            "'close-up macro photo of a honeybee on a yellow flower', "
            "'wide shot of Saturn V rocket launching at night with flames'. "
            "ENGLISH LENGTH: variants.en.full_narration must be long-form — "
            "aim 120-155 English words (never a short teaser); include hook, 3-4 developed beats with examples, "
            "and a strong closing line so the English voiceover is ~40-50 seconds."
        ),
        "segment_count": 5,
        "image_style_suffix": (
            ", photorealistic documentary photography, cinematic lighting, ultra detailed, "
            "8k, sharp focus, professional camera, National Geographic style, realistic textures, "
            "natural colors, depth of field, no text, no captions, no watermark, no logos"
        ),
        "image_negative_prompt": (
            "cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, "
            "stylized, flat colors, low quality, blurry, watermark, logo, text, signature, "
            "deformed, ugly, extra limbs, mutated"
        ),
        "topic_pool": [
            # Space & Universe
            "black holes", "neutron stars", "Mars mysteries", "Moon secrets", "exoplanets",
            "the Big Bang", "dark matter", "the asteroid belt", "Saturn's rings",
            "Jupiter's storms", "sounds in space", "dead stars", "parallel universes",
            "time dilation", "cosmic radiation", "space colonies", "SETI and alien signals",
            "the Voyager probe", "sun facts", "galaxy collisions", "space weather",
            "quantum physics weirdness", "multiverse theory", "wormholes",
            "the edge of the observable universe",
            # Animals & Nature
            "deep sea creatures", "parasites that control minds", "animal superpowers",
            "extinct animals", "animals that basically can't die",
            "venomous creatures of India", "crow intelligence", "the octopus brain",
            "the mantis shrimp", "tardigrades", "animal sleep habits", "migration mysteries",
            "camouflage masters", "animals that mourn their dead", "bioluminescence",
            "carnivorous plants", "fungi intelligence", "ant colonies", "whale communication",
            "spider silk science", "the immortal jellyfish", "animal self-medication",
            "dolphin language", "electric eels", "naked mole rats", "bird navigation",
            "snake facts", "insects you didn't know exist", "animals in Indian forests",
            "microorganisms living in your body",
            # Human Body & Psychology
            "brain illusions", "the science of sleep paralysis", "memory tricks",
            "the placebo effect", "pain tolerance", "human senses you didn't know about",
            "DNA secrets", "the gut-brain connection", "adrenaline effects",
            "the subconscious mind", "phobias explained", "the science of dreams",
            "body language secrets", "why we laugh", "human evolution oddities",
            "the science of aging", "near-death experiences", "hypnosis facts",
            "déjà vu explained", "emotional memory", "synesthesia", "muscle memory",
            "the fear response", "the science of addiction", "human body record breakers",
            # History & Civilizations
            "Ancient Egypt secrets", "dark facts about the Roman Empire", "lost civilizations",
            "medieval torture devices", "ancient Indian empires", "Mughal secrets",
            "forgotten inventions", "unknown facts about World War 2", "Cold War spy stories",
            "Greek myths debunked", "real Viking history", "the Aztec civilization",
            "the Indus Valley mystery", "the Maurya Empire", "the Chola naval empire",
            "the history of slavery", "ancient medicines", "the oldest cities on Earth",
            "ancient trade routes", "Genghis Khan facts", "the real Cleopatra",
            "Alexander the Great", "dark facts about British India",
            "untold partition of India stories", "ancient Chinese secrets",
            "the real life of samurai", "real pirate history", "the Byzantine Empire",
            "the Ottoman Empire", "ancient astronomy",
            # India Specific
            "India's unsolved mysteries", "cursed places in India",
            "unknown Indian inventions", "weird Indian laws",
            "dark stories from Indian mythology", "haunted forts of India",
            "untold Indian freedom fighters", "India's richest kings in history",
            "origin stories of Indian street food", "India's rarest animals",
            "Indian space program facts", "the engineering of ancient temples",
            "India's tribal cultures", "Bollywood dark secrets",
            "India's geographical oddities", "mysteries of Indian rivers",
            "Indian martial arts", "underground cities of India",
            "India's hottest and coldest places", "facts about Indian languages",
            # Money & Power
            "the richest people in history", "how billionaires think",
            "the dark side of corporations", "famous stock market crashes",
            "consequences of money printing", "heists gone wrong",
            "underground economies", "tax havens explained",
            "the richest countries in history", "failed currencies",
            "the gold standard", "dark stories from crypto", "the mafia economy",
            "war profiteering", "untold stories of India's richest businessmen",
            "how banks really work", "money psychology", "poverty traps",
            "famous economic collapses", "the dark side of diamonds",
            # Science & Tech
            "when AI went wrong", "internet dark secrets", "nuclear energy facts",
            "genetic engineering", "CRISPR experiments", "lab-grown meat",
            "deepfake technology", "social media algorithms", "dark web facts",
            "surveillance technology", "robot evolution", "battery technology secrets",
            "shocking climate science facts", "plastic in the human body",
            "microwave radiation", "5G facts vs myths", "quantum computing",
            "the history of bioweapons", "chemical reactions gone wrong",
            "future technology predictions",
            # Food & Substances
            "foods your brain is addicted to", "the dark side of sugar",
            "fast food secrets", "spices that changed history",
            "poisonous foods we eat daily", "the science of fermentation",
            "caffeine deep dive", "the science of alcohol",
            "the spiciest things on Earth", "food frauds worldwide",
            "ancient recipes still used today", "rare foods only the rich eat",
            "the history of Indian spices", "foods banned around the world",
            "GMO food facts",
            # Crime & Dark Secrets
            "unsolved murders", "serial killer psychology",
            "cults that shocked the world", "government experiments on humans",
            "corporate cover-ups", "art heists", "counterfeit economies",
            "organized crime facts", "famous prison escapes",
            "cold cases solved by DNA", "the biggest cyber crimes ever",
            "assassination plots", "whistleblower stories",
            "the dark side of Hollywood", "scams that fooled millions",
            "human trafficking networks", "drug cartel economics",
            "facts about corrupt governments", "identity theft stories",
            "history of con artists",
            # Wildcards
            "dreams that predicted the future", "coincidences too weird to be real",
            "things banned in other countries", "phobias with unpronounceable names",
            "world records that sound fake", "the science of optical illusions",
            "superstitions with real origins", "things that didn't exist 20 years ago",
            "urban legends debunked", "numbers with dark histories",
        ],
    },
    "hindi_myth": {
        "id": "hindi_myth",
        "label": "Hindi mythology & devotion Shorts (Ganesha → Shiva → … by IST day)",
        "topic_rotation": "myth",
        "language": "hi",
        "min_words": 100,
        # Edge TTS Hindi: Swara = warm female (common for katha / devotion). Override: hi-IN-MadhurNeural (male).
        "tts_voice": "hi-IN-SwaraNeural",
        "caption_font": "NotoSansDevanagari-Bold.ttf",
        "caption_font_name": "Noto Sans Devanagari",
        "yt_token_env": "YT_REFRESH_TOKEN_MYTH",
        "extra_yt_token_envs": ["YT_REFRESH_TOKEN_MYTH_2"],
        "groq_system_hint": (
            "You write respectful Hindi Shorts about Indian mythology, epics, and devotion — for a general audience. "
            "
    return sorted(PRESETS.keys())


def get_preset(channel_id: str) -> ChannelPreset:
    key = channel_id.strip().lower().replace("-", "_")
    if key not in PRESETS:
        raise KeyError(f"Unknown channel preset {channel_id!r}. Try: {', '.join(list_channel_ids())}")
    return PRESETS[key]
