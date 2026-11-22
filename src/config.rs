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
    #[serde(default = "default_char")]
    pub(crate) char: String,
    #[serde(default = "default_user")]
    pub(crate) user: String,
    #[serde(default = "default_them")]
    pub(crate) them: String,
    #[serde(default = "default_their")]
    pub(crate) their: String,
    #[serde(default = "default_self")]
    pub(crate) self_ref: String,
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
        text.replace("{char}", &self.persona.char)
            .replace("{user}", &self.persona.user)
            .replace("{them}", &self.persona.them)
            .replace("{their}", &self.persona.their)
            .replace("{self}", &self.persona.self_ref)
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

fn default_char() -> String {
    "mommy".into()
}

fn default_user() -> String {
    "good girl".into()
}

fn default_them() -> String {
    "her".into()
}

fn default_their() -> String {
    "her".into()
}

fn default_self() -> String {
    "herself".into()
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
