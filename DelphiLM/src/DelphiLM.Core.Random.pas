{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Core.Random;

interface

type
  TXorShift64Star = record
  private
    FState: UInt64;
  public
    procedure Initialize(const ASeed: UInt64);
    function NextUInt64: UInt64;
    function NextSingle: Single;
  end;

implementation

procedure TXorShift64Star.Initialize(const ASeed: UInt64);
begin
  FState := ASeed;
  if FState = 0 then
    FState := UInt64($9E3779B97F4A7C15);
end;

function TXorShift64Star.NextUInt64: UInt64;
begin
  FState := FState xor (FState shr 12);
  FState := FState xor (FState shl 25);
  FState := FState xor (FState shr 27);
  Result := FState * UInt64(2685821657736338717);
end;

function TXorShift64Star.NextSingle: Single;
const
  Scale: Double = 1.0 / 16777216.0;
begin
  Result := Single((NextUInt64 shr 40) * Scale);
end;

end.
