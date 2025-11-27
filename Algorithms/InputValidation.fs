module Algorithms.InputValidation

open System
open System.IO

/// Errors that can occur during input file resolution
type InputValidationError =
    | InputNotFound of path: string
    | FileListNotFound of path: string
    | NoMatchingFiles of pattern: string

    override this.ToString() =
        match this with
        | InputNotFound path -> $"Input directory not found: {path}"
        | FileListNotFound path -> $"File list not found: {path}"
        | NoMatchingFiles pattern -> $"No files found matching pattern: {pattern}"

/// Resolve input files from a pattern or file list
/// Supports:
/// - Glob patterns: "images/*.xisf"
/// - File lists: "@filelist.txt" (one file per line)
let resolveInputFiles (pattern: string) : Result<string[], InputValidationError> =
    if pattern.StartsWith("@") then
        // Read file list from text file
        let listPath = pattern.Substring(1)
        if not (File.Exists(listPath)) then
            Error (FileListNotFound listPath)
        else
            let files =
                File.ReadAllLines(listPath)
                |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
                |> Array.sort
            if files.Length = 0 then
                Error (NoMatchingFiles pattern)
            else
                Ok files
    else
        // Use glob pattern
        let inputDir =
            match Path.GetDirectoryName(pattern) with
            | null | "" -> "."
            | s -> s
        let globPattern = Path.GetFileName(pattern)

        if not (Directory.Exists inputDir) then
            Error (InputNotFound inputDir)
        else
            let files = Directory.GetFiles(inputDir, globPattern) |> Array.sort
            if files.Length = 0 then
                Error (NoMatchingFiles pattern)
            else
                Ok files
