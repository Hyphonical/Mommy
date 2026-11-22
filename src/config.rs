use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, Default)]
pub(crate) struct Config {
    #[serde(default)]
    pub(crate) persona: Persona,
    #[serde(default)]
    pub(crate) style: Style,
}

#[derive(Debug, Deserialize, Clone)]
pub(crate) struct Persona {
    #[serde(default = "default_char", deserialize_with = "one_or_many")]
    pub(crate) char: Vec<String>,
    #[serde(default = "default_user", deserialize_with = "one_or_many")]
    pub(crate) user: Vec<String>,
    #[serde(default = "default_them", deserialize_with = "one_or_many")]
    pub(crate) them: Vec<String>,
    #[serde(default = "default_their", deserialize_with = "one_or_many")]
    pub(crate) their: Vec<String>,
    #[serde(default = "default_self", deserialize_with = "one_or_many")]
    pub(crate) self_ref: Vec<String>,
}

impl Default for Persona {
    fn default() -> Self {
        Self {
            char: default_char(),
            user: default_user(),
            them: default_them(),
            their: default_their(),
            self_ref: default_self(),
        }
    }
}

#[derive(Debug, Deserialize, Default, Clone)]
pub(crate) struct Style {
    pub(crate) color: Option<String>,
}

impl Config {
    pub(crate) fn load() -> Self {
        let path = if let Ok(value) = std::env::var("MOMMY_CONFIG") {
            std::path::PathBuf::from(value)
        } else if let Some(home) = crate::home_dir() {
            home.join(".mommy").join("config.toml")
        } else {
            return Self::default();
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        match toml::from_str(&raw) {
            Ok(config) => config,
            Err(error) => {
                eprintln!("mommy: bad config {}: {error}", path.display());
                Self::default()
            }
        }
    }

    pub(crate) fn substitute(&self, text: &str) -> String {
        let mut rng = crate::rng::Rng::new();
        let char = pick_one(&self.persona.char, "mommy", &mut rng);
        let user = pick_one(&self.persona.user, "good girl", &mut rng);
        let them = pick_one(&self.persona.them, "her", &mut rng);
        let their = pick_one(&self.persona.their, "her", &mut rng);
        let self_ref = pick_one(&self.persona.self_ref, "herself", &mut rng);
        text.replace("{char}", &char)
            .replace("{user}", &user)
            .replace("{them}", &them)
            .replace("{their}", &their)
            .replace("{self}", &self_ref)
    }

    pub(crate) fn colorize(&self, text: &str) -> String {
        let Some(hex) = &self.style.color else {
            return text.to_owned();
        };
        if std::env::var("NO_COLOR").is_ok_and(|value| !value.is_empty()) {
            return text.to_owned();
        }
        let Some((red, green, blue)) = parse_hex(hex) else {
            eprintln!("mommy: bad color {hex:?}, ignoring");
            return text.to_owned();
        };
        let code = if std::env::var("COLORTERM")
            .is_ok_and(|value| value == "truecolor" || value == "24bit")
        {
            format!("38;2;{red};{green};{blue}")
        } else if std::env::var("TERM").is_ok_and(|value| value.contains("256")) {
            format!("38;5;{}", ansi256(red, green, blue))
        } else {
            return text.to_owned();
        };
        format!("\x1b[{code}m{text}\x1b[0m")
    }
}

fn one_or_many<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(value) => Ok(vec![value]),
        OneOrMany::Many(values) => Ok(values),
    }
}

fn pick_one(values: &[String], fallback: &str, rng: &mut crate::rng::Rng) -> String {
    let clean: Vec<&str> = values
        .iter()
        .map(String::as_str)
        .filter(|value| !value.trim().is_empty())
        .collect();
    if clean.is_empty() {
        return fallback.to_owned();
    }
    clean[(rng.next_u64() as usize) % clean.len()].to_owned()
}

fn default_char() -> Vec<String> {
    vec!["mommy".into()]
}

fn default_user() -> Vec<String> {
    vec!["good girl".into()]
}

fn default_them() -> Vec<String> {
    vec!["her".into()]
}

fn default_their() -> Vec<String> {
    vec!["her".into()]
}

fn default_self() -> Vec<String> {
    vec!["herself".into()]
}

fn parse_hex(value: &str) -> Option<(u8, u8, u8)> {
    let value = value.strip_prefix('#')?;
    if value.len() != 6 {
        return None;
    }
    let red = u8::from_str_radix(&value[0..2], 16).ok()?;
    let green = u8::from_str_radix(&value[2..4], 16).ok()?;
    let blue = u8::from_str_radix(&value[4..6], 16).ok()?;
    Some((red, green, blue))
}

fn ansi256(red: u8, green: u8, blue: u8) -> u16 {
    let quantize = |channel: u8| (u16::from(channel) / 51).min(5);
    16 + 36 * quantize(red) + 6 * quantize(green) + quantize(blue)
}
