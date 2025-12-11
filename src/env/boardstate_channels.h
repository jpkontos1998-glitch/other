#pragma once

#include <array>
#include <string>
#include <vector>

// Helper function to generate channel descriptions
inline std::vector<std::string> GenerateBoardstateChannelDescriptions() {
    std::vector<std::string> descriptions;
    const std::vector<std::string> pieces = {"spy", "scout", "miner", "sergeant", "lieutenant", 
                                           "captain", "major", "colonel", "general", "marshal", 
                                           "flag", "bomb"};
    const std::vector<std::string> pieces_no_flag = {"spy", "scout", "miner", "sergeant", "lieutenant", 
                                                   "captain", "major", "colonel", "general", "marshal"};
    const std::vector<std::string> pieces_with_unknown = {"spy", "scout", "miner", "sergeant", "lieutenant", 
                                                        "captain", "major", "colonel", "general", "marshal", 
                                                        "unknown"};
    const std::vector<std::string> pieces_with_extras = {"spy", "scout", "miner", "sergeant", "lieutenant", 
                                                       "captain", "major", "colonel", "general", "marshal", 
                                                       "bomb", "empty", "unknown"};
    const std::vector<std::string> death_reasons = {"attacked_visible_stronger", "attacked_visible_tie", 
                                                  "attacked_hidden", "visible_defended_weaker", 
                                                  "visible_defended_tie", "hidden_defended"};

    // Our pieces
    for (const auto& piece : pieces) {
        descriptions.push_back("our_" + piece);
    }

    // Their piece probabilities
    for (const auto& piece : pieces) {
        descriptions.push_back("their_" + piece + "_prob");
    }

    // Our piece probabilities
    for (const auto& piece : pieces) {
        descriptions.push_back("our_" + piece + "_prob");
    }

    // Basic state
    descriptions.push_back("our_hidden_bool");
    descriptions.push_back("their_hidden_bool");
    descriptions.push_back("empty_bool");
    descriptions.push_back("our_moved_bool");
    descriptions.push_back("their_moved_bool");
    descriptions.push_back("max_num_moves_frac");
    descriptions.push_back("max_num_moves_between_attacks_frac");

    // Threat/evasion/adjacency
    for (const auto& piece : pieces_with_unknown) {
        descriptions.push_back("we_threatened_" + piece);
    }
    for (const auto& piece : pieces_with_unknown) {
        descriptions.push_back("we_evaded_" + piece);
    }
    for (const auto& piece : pieces_with_unknown) {
        descriptions.push_back("we_actively_adj_" + piece);
    }
    for (const auto& piece : pieces_with_unknown) {
        descriptions.push_back("they_threatened_" + piece);
    }
    for (const auto& piece : pieces_with_unknown) {
        descriptions.push_back("they_evaded_" + piece);
    }
    for (const auto& piece : pieces_with_unknown) {
        descriptions.push_back("they_actively_adj_" + piece);
    }

    // Dead pieces
    for (const auto& piece : pieces_no_flag) {
        descriptions.push_back("our_dead_" + piece);
    }
    descriptions.push_back("our_dead_bomb");
    for (const auto& piece : pieces_no_flag) {
        descriptions.push_back("their_dead_" + piece);
    }
    descriptions.push_back("their_dead_bomb");

    // Death status
    for (const auto& reason : death_reasons) {
        for (const auto& piece : pieces_no_flag) {
            descriptions.push_back("our_deathstatus_" + reason + "_" + piece);
        }
    }
    for (const auto& reason : death_reasons) {
        for (const auto& piece : pieces_no_flag) {
            descriptions.push_back("their_deathstatus_" + reason + "_" + piece);
        }
    }

    // Protection
    for (const auto& piece : pieces_with_extras) {
        descriptions.push_back("our_protected_" + piece);
    }
    for (const auto& piece : pieces_with_extras) {
        descriptions.push_back("our_protected_against_" + piece);
    }
    for (const auto& piece : pieces_with_extras) {
        descriptions.push_back("our_was_protected_by_" + piece);
    }
    for (const auto& piece : pieces_with_extras) {
        descriptions.push_back("our_was_protected_against_" + piece);
    }
    for (const auto& piece : pieces_with_extras) {
        descriptions.push_back("their_protected_" + piece);
    }
    for (const auto& piece : pieces_with_extras) {
        descriptions.push_back("their_protected_against_" + piece);
    }
    for (const auto& piece : pieces_with_extras) {
        descriptions.push_back("their_was_protected_by_" + piece);
    }
    for (const auto& piece : pieces_with_extras) {
        descriptions.push_back("their_was_protected_against_" + piece);
    }

    return descriptions;
}

// Base channel descriptions that are always present
const std::vector<std::string> BOARDSTATE_CHANNEL_DESCRIPTIONS = GenerateBoardstateChannelDescriptions();