use std::convert::From;
use std::vec::Vec;

pub fn vs2vstr(ori: Vec<&str>) -> Vec<String> {
    ori.iter().map(|&s| s.to_string()).collect()
}

pub fn as2vstr(ori: &[&str]) -> Vec<String> {
    ori.iter().map(|&s| s.to_string()).collect()
}

#[test]
fn test_vecstr2string() {
    let ops = vec!["5", "2", "C", "D", "+"];
    assert_eq!(
        vec![
            "5".to_string(),
            "2".to_string(),
            "C".to_string(),
            "D".to_string(),
            "+".to_string()
        ],
        vs2vstr(ops)
    );
}

#[test]
fn test_arraystr2vecstring() {
    let ops = ["5", "2", "C", "D", "+"];
    assert_eq!(
        vec![
            "5".to_string(),
            "2".to_string(),
            "C".to_string(),
            "D".to_string(),
            "+".to_string()
        ],
        as2vstr(&ops)
    );
}
