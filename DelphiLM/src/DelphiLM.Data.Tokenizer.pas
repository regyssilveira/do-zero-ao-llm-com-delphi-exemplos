{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Data.Tokenizer;

interface

uses
  System.Generics.Collections,
  System.SysUtils;

type
  EDelphiLMTokenizerError = class(Exception);

  TCharacterTokenizer = class
  private
    FTokenToCharacter: TArray<Char>;
    FCharacterToToken: TDictionary<Char, Integer>;
  public
    constructor Create(const ACorpus: string);
    destructor Destroy; override;
    class function Normalize(const AText: string): string; static;
    function Encode(const AText: string): TArray<Integer>;
    function Decode(const ATokens: TArray<Integer>): string;
    function VocabularySize: Integer;
    function CharacterForToken(const AToken: Integer): Char;
  end;

implementation

uses
  System.Generics.Defaults,
  Winapi.Windows;

constructor TCharacterTokenizer.Create(const ACorpus: string);
var
  CharacterSet: TDictionary<Char, Byte>;
  Characters: TList<Char>;
  CharacterValue: Char;
  NormalizedCorpus: string;
  Token: Integer;
begin
  inherited Create;
  FCharacterToToken := TDictionary<Char, Integer>.Create;
  CharacterSet := TDictionary<Char, Byte>.Create;
  Characters := TList<Char>.Create;
  try
    NormalizedCorpus := Normalize(ACorpus);
    if NormalizedCorpus.IsEmpty then
      raise EDelphiLMTokenizerError.Create('O corpus não pode estar vazio.');

    for CharacterValue in NormalizedCorpus do
      if not CharacterSet.ContainsKey(CharacterValue) then
      begin
        CharacterSet.Add(CharacterValue, 0);
        Characters.Add(CharacterValue);
      end;

    Characters.Sort(TComparer<Char>.Default);
    SetLength(FTokenToCharacter, Characters.Count);
    for Token := 0 to Characters.Count - 1 do
    begin
      FTokenToCharacter[Token] := Characters[Token];
      FCharacterToToken.Add(Characters[Token], Token);
    end;
  finally
    Characters.Free;
    CharacterSet.Free;
  end;
end;

destructor TCharacterTokenizer.Destroy;
begin
  FCharacterToToken.Free;
  inherited;
end;

class function TCharacterTokenizer.Normalize(const AText: string): string;
var
  RequiredLength: Integer;
  WrittenLength: Integer;
begin
  if AText.IsEmpty then
    Exit('');

  RequiredLength := NormalizeString(NormalizationC, PChar(AText),
    Length(AText), nil, 0);
  if RequiredLength <= 0 then
    RaiseLastOSError;

  SetLength(Result, RequiredLength);
  WrittenLength := NormalizeString(NormalizationC, PChar(AText),
    Length(AText), PChar(Result), RequiredLength);
  if WrittenLength <= 0 then
    RaiseLastOSError;
  SetLength(Result, WrittenLength);
end;

function TCharacterTokenizer.Encode(const AText: string): TArray<Integer>;
var
  CharacterValue: Char;
  Index: Integer;
  NormalizedText: string;
  Token: Integer;
begin
  NormalizedText := Normalize(AText);
  SetLength(Result, Length(NormalizedText));
  Index := 0;
  for CharacterValue in NormalizedText do
  begin
    if not FCharacterToToken.TryGetValue(CharacterValue, Token) then
      raise EDelphiLMTokenizerError.CreateFmt(
        'Caractere U+%.4X não pertence ao vocabulário.',
        [Ord(CharacterValue)]);
    Result[Index] := Token;
    Inc(Index);
  end;
end;

function TCharacterTokenizer.Decode(
  const ATokens: TArray<Integer>): string;
var
  Index: Integer;
  Token: Integer;
begin
  SetLength(Result, Length(ATokens));
  for Index := 0 to High(ATokens) do
  begin
    Token := ATokens[Index];
    if (Token < 0) or (Token >= Length(FTokenToCharacter)) then
      raise EDelphiLMTokenizerError.CreateFmt(
        'Token %d está fora do vocabulário.', [Token]);
    Result[Index + 1] := FTokenToCharacter[Token];
  end;
end;

function TCharacterTokenizer.VocabularySize: Integer;
begin
  Result := Length(FTokenToCharacter);
end;

function TCharacterTokenizer.CharacterForToken(
  const AToken: Integer): Char;
begin
  if (AToken < 0) or (AToken >= Length(FTokenToCharacter)) then
    raise EDelphiLMTokenizerError.CreateFmt(
      'Token %d está fora do vocabulário.', [AToken]);
  Result := FTokenToCharacter[AToken];
end;

end.
