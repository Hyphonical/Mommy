/// ChatML prompt fragments for the single fixed template.
///
/// There is deliberately no system turn: the LoRA will bake the behaviour in
/// from training on this exact template, so inference must reproduce it
/// verbatim. Keep these fragments identical to the training format.
pub(crate) const USER_START: &str = "<|im_start|>user\n";
pub(crate) const ASSISTANT_START: &str = "<|im_end|>\n<|im_start|>assistant\n";

pub(crate) const STDOUT_HEADER: &str = "--- stdout (tail) ---\n";
pub(crate) const STDERR_HEADER: &str = "--- stderr (tail) ---\n";

pub(crate) fn status_line(status: i32, command: &str, duration: &str) -> String {
    format!("status={status} cmd=\"{command}\" dur={duration}\n")
}

pub(crate) fn format_duration(elapsed: std::time::Duration) -> String {
    let millis = elapsed.as_millis();
    if millis < 1000 {
        return format!("{millis}ms");
    }
    let tenths = millis / 100;
    if tenths.is_multiple_of(10) {
        format!("{}s", tenths / 10)
    } else {
        format!("{}.{}s", tenths / 10, tenths % 10)
    }
}
